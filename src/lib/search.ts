import type { Store } from './store.js';
import type { Embedder } from './embedder.js';
import type { SearchResult } from './types.js';

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
