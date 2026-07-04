import type { Store } from './store.js';
import type { Embedder } from './embedder.js';
import type { SearchResult } from './types.js';
import { rrfFuse, toFtsQuery } from './fusion.js';

// Graph-aware re-ranking: each result's score is multiplied by
// 1 + weight * normalizedPageRank, so well-connected (central) notes rank
// higher among comparable matches. Env-overridable; 0 disables it.
function graphBoostWeight(): number {
  const raw = Number(process.env.KG_GRAPH_BOOST_WEIGHT);
  return Number.isFinite(raw) && raw >= 0 ? raw : 0.25;
}

export class Search {
  constructor(
    private store: Store,
    private embedder: Embedder,
  ) {}

  async semantic(query: string, limit = 20): Promise<SearchResult[]> {
    const queryEmbedding = await this.embedder.embed(query);
    return this.boostByCentrality(this.store.searchVector(queryEmbedding, limit));
  }

  fulltext(query: string, limit = 20): SearchResult[] {
    return this.boostByCentrality(this.store.searchFullText(query).slice(0, limit));
  }

  /**
   * Hybrid search: RRF-fuse the dense (vector) and lexical (FTS5) channels,
   * then apply the centrality boost to the fused ranking. Rank-based fusion
   * because cosine distance and BM25 rank are not comparable scales. The
   * lexical channel is best-effort — a query FTS5 cannot parse (or an empty
   * token set) degrades to dense-only rather than failing the search.
   */
  async hybrid(query: string, limit = 20): Promise<SearchResult[]> {
    const queryEmbedding = await this.embedder.embed(query);
    const dense = this.store.searchVector(queryEmbedding, Math.max(limit * 2, 20));

    let lexical: SearchResult[] = [];
    const ftsQuery = toFtsQuery(query);
    if (ftsQuery !== '') {
      try {
        lexical = this.store.searchFullText(ftsQuery);
      } catch {
        lexical = [];
      }
    }

    const fused = rrfFuse([
      dense.map(r => r.nodeId),
      lexical.map(r => r.nodeId),
    ]);

    // Prefer the lexical excerpt (carries FTS5 match highlighting) when a node
    // was hit by both channels.
    const byId = new Map<string, SearchResult>();
    for (const r of lexical) byId.set(r.nodeId, r);
    for (const r of dense) {
      if (!byId.has(r.nodeId)) byId.set(r.nodeId, r);
    }

    const results: SearchResult[] = [];
    for (const [nodeId, score] of fused) {
      const base = byId.get(nodeId);
      if (base) results.push({ ...base, score });
    }
    return this.boostByCentrality(results).slice(0, limit);
  }

  /**
   * Re-rank retrieved results by a saturating PageRank boost. Operates only on
   * the already-retrieved top-k, so it re-orders rather than changing recall.
   */
  private boostByCentrality(results: SearchResult[]): SearchResult[] {
    const weight = graphBoostWeight();
    if (weight <= 0 || results.length === 0) return results;
    const centrality = this.store.getCentralityMap(results.map(r => r.nodeId));
    return results
      .map(r => ({ ...r, score: r.score * (1 + weight * (centrality.get(r.nodeId) ?? 0)) }))
      .sort((a, b) => b.score - a.score);
  }
}
