/**
 * Rank fusion for hybrid search. The dense (sqlite-vec cosine) and lexical
 * (FTS5 BM25) channels score on incomparable scales, so fusion is rank-based:
 * Reciprocal Rank Fusion needs only each channel's ordering. Pure functions,
 * no I/O — unit-testable in isolation from the store and embedder.
 */

/**
 * Reciprocal Rank Fusion over ranked id lists. Score for an id is the sum of
 * 1/(k + rank) across the lists it appears in, with 1-based rank. Ids in
 * multiple lists accumulate — agreement between channels outranks a single
 * channel's top hit.
 */
export function rrfFuse(rankedLists: string[][], k = 60): Map<string, number> {
  const out = new Map<string, number>();
  for (const list of rankedLists) {
    list.forEach((id, idx) => {
      out.set(id, (out.get(id) ?? 0) + 1 / (k + idx + 1));
    });
  }
  return out;
}

/**
 * Sanitize a free-text query into a safe FTS5 MATCH expression: extract
 * unicode word tokens, double-quote each (quoting makes FTS5 treat operators
 * like OR/NOT/* as plain strings), and join with OR for recall — the dense
 * channel handles precision, so partial lexical matches are wanted. Returns
 * '' when no tokens survive; callers skip the lexical channel then.
 */
export function toFtsQuery(query: string): string {
  const tokens = query.match(/[\p{L}\p{N}_]+/gu) ?? [];
  return tokens.map(t => `"${t}"`).join(' OR ');
}
