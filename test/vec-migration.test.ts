import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { Store } from '../src/lib/store.js';
import { resolveEmbedDim } from '../src/lib/config.js';

function vecTableSql(store: Store): string {
  const row = store.db.prepare(
    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'nodes_vec'"
  ).get() as { sql: string };
  return row.sql;
}

describe('resolveEmbedDim', () => {
  const original = process.env.KG_EMBED_DIM;

  afterEach(() => {
    if (original === undefined) delete process.env.KG_EMBED_DIM;
    else process.env.KG_EMBED_DIM = original;
  });

  it('defaults to 768', () => {
    delete process.env.KG_EMBED_DIM;
    expect(resolveEmbedDim()).toBe(768);
  });

  it('honors KG_EMBED_DIM', () => {
    process.env.KG_EMBED_DIM = '384';
    expect(resolveEmbedDim()).toBe(384);
  });

  it('falls back to default on invalid values', () => {
    for (const bad of ['0', '-5', '1.5', 'abc', '']) {
      process.env.KG_EMBED_DIM = bad;
      expect(resolveEmbedDim()).toBe(768);
    }
  });
});

describe('nodes_vec dimension migration', () => {
  let dir: string;
  let dbPath: string;
  const originalDim = process.env.KG_EMBED_DIM;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'kg-vec-migration-'));
    dbPath = join(dir, 'kg.db');
  });

  afterEach(() => {
    if (originalDim === undefined) delete process.env.KG_EMBED_DIM;
    else process.env.KG_EMBED_DIM = originalDim;
    rmSync(dir, { recursive: true, force: true });
  });

  it('creates the vec table at the configured dimension', () => {
    delete process.env.KG_EMBED_DIM;
    const store = new Store(dbPath);
    expect(vecTableSql(store)).toContain('float[768]');
    store.close();
  });

  it('rebuilds the vec table and clears sync when the dimension changes', () => {
    // Simulate an old 384-dim database with indexed content.
    process.env.KG_EMBED_DIM = '384';
    const oldStore = new Store(dbPath);
    oldStore.upsertNode({ id: 'a.md', title: 'Alpha', content: 'alpha content', frontmatter: {} });
    oldStore.upsertEmbedding('a.md', new Float32Array(384).fill(0.1));
    oldStore.upsertSync('a.md', 12345);
    oldStore.close();

    // Reopen at the new 768-dim default.
    delete process.env.KG_EMBED_DIM;
    const store = new Store(dbPath);

    expect(vecTableSql(store)).toContain('float[768]');
    // Vector index rebuilt empty; sync cleared so next index re-embeds all.
    const vecCount = store.db.prepare('SELECT COUNT(*) as cnt FROM nodes_vec').get() as { cnt: number };
    expect(vecCount.cnt).toBe(0);
    expect(store.getSyncMtime('a.md')).toBeUndefined();
    // Node content itself is preserved.
    expect(store.getNode('a.md')?.title).toBe('Alpha');

    // Writes and searches at the new dimension work.
    store.upsertEmbedding('a.md', new Float32Array(768).fill(0.1));
    const results = store.searchVector(new Float32Array(768).fill(0.1), 5);
    expect(results.length).toBe(1);
    expect(results[0].nodeId).toBe('a.md');
    store.close();
  });

  it('leaves the vec table and sync intact when the dimension matches', () => {
    delete process.env.KG_EMBED_DIM;
    const first = new Store(dbPath);
    first.upsertNode({ id: 'a.md', title: 'Alpha', content: 'alpha content', frontmatter: {} });
    first.upsertEmbedding('a.md', new Float32Array(768).fill(0.1));
    first.upsertSync('a.md', 12345);
    first.close();

    const store = new Store(dbPath);
    const vecCount = store.db.prepare('SELECT COUNT(*) as cnt FROM nodes_vec').get() as { cnt: number };
    expect(vecCount.cnt).toBe(1);
    expect(store.getSyncMtime('a.md')).toBe(12345);
    store.close();
  });
});
