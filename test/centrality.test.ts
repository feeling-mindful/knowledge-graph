import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Store } from '../src/lib/store.js';
import { Embedder } from '../src/lib/embedder.js';
import { Search } from '../src/lib/search.js';

describe('centrality store', () => {
  let store: Store;

  beforeEach(() => {
    store = new Store(':memory:');
  });

  afterEach(() => {
    store.close();
  });

  it('upserts and reads a node pagerank', () => {
    store.upsertCentrality('a.md', 0.8);
    expect(store.getCentrality('a.md')).toBeCloseTo(0.8, 6);
  });

  it('returns 0 for an unknown node', () => {
    expect(store.getCentrality('missing.md')).toBe(0);
  });

  it('overwrites on conflict', () => {
    store.upsertCentrality('a.md', 0.2);
    store.upsertCentrality('a.md', 0.9);
    expect(store.getCentrality('a.md')).toBeCloseTo(0.9, 6);
  });

  it('batch-fetches a centrality map', () => {
    store.upsertCentrality('a.md', 0.5);
    store.upsertCentrality('b.md', 1.0);
    const map = store.getCentralityMap(['a.md', 'b.md', 'c.md']);
    expect(map.get('a.md')).toBeCloseTo(0.5, 6);
    expect(map.get('b.md')).toBeCloseTo(1.0, 6);
    expect(map.has('c.md')).toBe(false);
    expect(store.getCentralityMap([]).size).toBe(0);
  });

  it('clears centrality', () => {
    store.upsertCentrality('a.md', 0.5);
    store.clearCentrality();
    expect(store.getCentrality('a.md')).toBe(0);
  });

  it('removes a node\'s centrality row when the node is deleted', () => {
    store.upsertNode({ id: 'a.md', title: 'A', content: 'body', frontmatter: {} });
    store.upsertCentrality('a.md', 0.7);
    store.deleteNode('a.md');
    expect(store.getCentrality('a.md')).toBe(0);
  });
});

describe('search centrality boost (fulltext)', () => {
  let store: Store;
  let search: Search;
  let prev: string | undefined;

  beforeEach(() => {
    prev = process.env.KG_GRAPH_BOOST_WEIGHT;
    store = new Store(':memory:');
    // fulltext path does not call the embedder.
    search = new Search(store, new Embedder());
    // Two nodes that both match "alpha"; central.md is more connected.
    store.upsertNode({ id: 'plain.md', title: 'alpha plain', content: 'alpha topic notes', frontmatter: {} });
    store.upsertNode({ id: 'central.md', title: 'alpha central', content: 'alpha topic notes', frontmatter: {} });
  });

  afterEach(() => {
    store.close();
    if (prev === undefined) delete process.env.KG_GRAPH_BOOST_WEIGHT;
    else process.env.KG_GRAPH_BOOST_WEIGHT = prev;
  });

  it('promotes a high-centrality node above an equal-text peer', () => {
    process.env.KG_GRAPH_BOOST_WEIGHT = '50';
    store.upsertCentrality('central.md', 1.0);
    store.upsertCentrality('plain.md', 0.0);

    const results = search.fulltext('alpha');

    expect(results.map(r => r.nodeId)).toEqual(['central.md', 'plain.md']);
  });

  it('leaves order unchanged when the boost weight is 0', () => {
    process.env.KG_GRAPH_BOOST_WEIGHT = '0';
    store.upsertCentrality('central.md', 1.0);

    const before = store.searchFullText('alpha').slice(0, 20).map(r => r.nodeId);
    const after = search.fulltext('alpha').map(r => r.nodeId);

    expect(after).toEqual(before);
  });
});
