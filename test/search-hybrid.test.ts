import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Store } from '../src/lib/store.js';
import { Embedder } from '../src/lib/embedder.js';
import { Search } from '../src/lib/search.js';

describe('Search.hybrid', () => {
  let store: Store;
  let embedder: Embedder;
  let search: Search;

  beforeAll(async () => {
    store = new Store(':memory:');
    embedder = new Embedder();
    await embedder.init();
    search = new Search(store, embedder);

    const nodes = [
      { id: 'graph.md', title: 'Graph Theory', content: 'Study of mathematical structures used to model pairwise relations between objects', frontmatter: {} },
      { id: 'cake.md', title: 'Chocolate Cake', content: 'A delicious dessert made with cocoa powder and sugar', frontmatter: {} },
      { id: 'network.md', title: 'Network Analysis', content: 'Analysis of graph structures in social networks', frontmatter: {} },
    ];

    for (const node of nodes) {
      store.upsertNode(node);
      const text = Embedder.buildEmbeddingText(node.title, [], node.content);
      const embedding = await embedder.embed(text);
      store.upsertEmbedding(node.id, embedding);
    }
  }, 60000);

  afterAll(async () => {
    store.close();
    await embedder.dispose();
  });

  it('ranks nodes hit by both channels above single-channel hits', async () => {
    // "graph" is a literal FTS hit AND semantically close for graph.md/network.md;
    // cake.md matches neither channel strongly.
    const results = await search.hybrid('graph structures');
    expect(results.length).toBeGreaterThan(0);
    const ids = results.map(r => r.nodeId);
    const cakeIdx = ids.indexOf('cake.md');
    for (const id of ['graph.md', 'network.md']) {
      const idx = ids.indexOf(id);
      expect(idx).toBeGreaterThanOrEqual(0);
      if (cakeIdx !== -1) expect(idx).toBeLessThan(cakeIdx);
    }
  });

  it('does not throw on FTS5 syntax in the query', async () => {
    const results = await search.hybrid('what\'s "this"? (dessert with cocoa) OR NOT*');
    // Semantic channel still contributes even if lexical parsing were hostile.
    expect(results.length).toBeGreaterThan(0);
  });

  it('respects the limit', async () => {
    const results = await search.hybrid('graph', 1);
    expect(results.length).toBe(1);
  });

  it('falls back to lexical hits for content without embeddings', async () => {
    store.upsertNode({ id: 'zebra.md', title: 'Zebra Facts', content: 'zebrastripe patterns are unique per animal', frontmatter: {} });
    // No upsertEmbedding — dense channel cannot see this node.
    const results = await search.hybrid('zebrastripe');
    expect(results.map(r => r.nodeId)).toContain('zebra.md');
  });

  it('orders by fused score even with the centrality boost disabled', async () => {
    const original = process.env.KG_GRAPH_BOOST_WEIGHT;
    process.env.KG_GRAPH_BOOST_WEIGHT = '0';
    try {
      const results = await search.hybrid('graph structures');
      const scores = results.map(r => r.score);
      expect(scores).toEqual([...scores].sort((a, b) => b - a));
      // Both-channel hits must still outrank cake.md with the boost off.
      const ids = results.map(r => r.nodeId);
      const cakeIdx = ids.indexOf('cake.md');
      if (cakeIdx !== -1) {
        expect(ids.indexOf('graph.md')).toBeLessThan(cakeIdx);
      }
    } finally {
      if (original === undefined) delete process.env.KG_GRAPH_BOOST_WEIGHT;
      else process.env.KG_GRAPH_BOOST_WEIGHT = original;
    }
  });

  it('clamps invalid limits to the default instead of misbehaving', async () => {
    for (const bad of [-3, 0, NaN]) {
      const results = await search.hybrid('graph', bad);
      expect(results.length).toBeGreaterThan(0);
      expect(results.length).toBeLessThanOrEqual(20);
    }
  });
});

describe('Store.searchFullText limit', () => {
  it('returns more than 20 rows when asked (hybrid candidate depth)', () => {
    const store = new Store(':memory:');
    for (let i = 0; i < 30; i++) {
      store.upsertNode({ id: `n${i}.md`, title: `Note ${i}`, content: 'shared keyword corpus entry', frontmatter: {} });
    }
    expect(store.searchFullText('keyword', 30).length).toBe(30);
    expect(store.searchFullText('keyword').length).toBe(20);
    store.close();
  });

  it('clamps invalid limits at the storage boundary (MCP/CLI can pass NaN or <= 0)', () => {
    const store = new Store(':memory:');
    for (let i = 0; i < 25; i++) {
      store.upsertNode({ id: `n${i}.md`, title: `Note ${i}`, content: 'shared keyword corpus entry', frontmatter: {} });
    }
    for (const bad of [NaN, 0, -5, 2.7]) {
      expect(store.searchFullText('keyword', bad).length).toBeLessThanOrEqual(20);
      expect(store.searchFullText('keyword', bad).length).toBeGreaterThan(0);
    }
    // searchVector must not throw on a bad k either.
    store.upsertEmbedding('n0.md', new Float32Array(768).fill(0.1));
    for (const bad of [NaN, 0, -5]) {
      expect(() => store.searchVector(new Float32Array(768).fill(0.1), bad)).not.toThrow();
    }
    store.close();
  });
});
