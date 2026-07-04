import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync, unlinkSync, mkdirSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { Store } from '../src/lib/store.js';
import type { Embedder } from '../src/lib/embedder.js';
import { IndexPipeline } from '../src/lib/index-pipeline.js';
import { resolveEmbedDim } from '../src/lib/config.js';

function vecTableSql(store: Store): string {
  const row = store.db.prepare(
    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'nodes_vec'"
  ).get() as { sql: string };
  return row.sql;
}

function vecRowCount(store: Store): number {
  const row = store.db.prepare('SELECT COUNT(*) as cnt FROM nodes_vec').get() as { cnt: number };
  return row.cnt;
}

/** Deterministic fake embedder producing vectors of a fixed dimension. */
function fakeEmbedder(dim: number): Embedder {
  return {
    embed: async () => new Float32Array(dim).fill(0.5),
  } as unknown as Embedder;
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

describe('nodes_vec dimension self-healing', () => {
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

  function makeLegacy384Db(): void {
    process.env.KG_EMBED_DIM = '384';
    const legacy = new Store(dbPath);
    legacy.upsertNode({ id: 'a.md', title: 'Alpha', content: 'alpha content', frontmatter: {} });
    legacy.upsertEmbedding('a.md', new Float32Array(384).fill(0.1));
    legacy.upsertSync('a.md', 12345);
    legacy.upsertSync('deleted-before-upgrade.md', 999);
    legacy.close();
    delete process.env.KG_EMBED_DIM;
  }

  it('creates the vec table at the configured dimension on a fresh DB', () => {
    delete process.env.KG_EMBED_DIM;
    const store = new Store(dbPath);
    expect(vecTableSql(store)).toContain('float[768]');
    store.close();
  });

  it('opening a legacy DB does NOT destroy the index (config alone never rebuilds)', () => {
    makeLegacy384Db();
    const store = new Store(dbPath); // default env resolves to 768
    expect(vecTableSql(store)).toContain('float[384]');
    expect(vecRowCount(store)).toBe(1);
    expect(store.getSyncMtime('a.md')).toBe(12345);
    store.close();
  });

  it('writing an embedding of a new dimension rebuilds the table and zeroes sync mtimes', () => {
    makeLegacy384Db();
    const store = new Store(dbPath);

    // The exact write that used to throw SqliteError: Dimension mismatch.
    store.upsertEmbedding('a.md', new Float32Array(768).fill(0.2));

    expect(vecTableSql(store)).toContain('float[768]');
    expect(vecRowCount(store)).toBe(1);
    const results = store.searchVector(new Float32Array(768).fill(0.2), 5);
    expect(results.map(r => r.nodeId)).toEqual(['a.md']);

    // Sync mtimes zeroed → every file re-indexes next pass; paths preserved →
    // deleted-file reconciliation still sees notes removed before the rebuild.
    expect(store.getSyncMtime('a.md')).toBe(0);
    expect(store.getAllSyncPaths()).toContain('deleted-before-upgrade.md');
    expect(store.getNode('a.md')?.title).toBe('Alpha');
    store.close();
  });

  it('searchVector degrades to empty results on dimension mismatch instead of throwing', () => {
    makeLegacy384Db();
    const store = new Store(dbPath);
    const results = store.searchVector(new Float32Array(768).fill(0.1), 5);
    expect(results).toEqual([]);
    // The 384-dim index is untouched — search must never trigger a rebuild.
    expect(vecTableSql(store)).toContain('float[384]');
    expect(vecRowCount(store)).toBe(1);
    store.close();
  });

  it('matching dimension leaves vec rows and sync mtimes intact', () => {
    delete process.env.KG_EMBED_DIM;
    const first = new Store(dbPath);
    first.upsertNode({ id: 'a.md', title: 'Alpha', content: 'alpha content', frontmatter: {} });
    first.upsertEmbedding('a.md', new Float32Array(768).fill(0.1));
    first.upsertSync('a.md', 12345);
    first.close();

    const store = new Store(dbPath);
    store.upsertEmbedding('a.md', new Float32Array(768).fill(0.3));
    expect(vecRowCount(store)).toBe(1);
    expect(store.getSyncMtime('a.md')).toBe(12345);
    store.close();
  });

  it('rejects invalid dimensions', () => {
    const store = new Store(dbPath);
    for (const bad of [0, -1, 1.5, NaN]) {
      expect(() => store.ensureVecDim(bad)).toThrow('Invalid embedding dimension');
    }
    store.close();
  });

  it('index pipeline probe re-embeds the whole vault and reconciles deletions after a model change', async () => {
    const vaultPath = join(dir, 'vault');
    mkdirSync(vaultPath);
    writeFileSync(join(vaultPath, 'a.md'), '# Alpha\n\nalpha content\n');
    writeFileSync(join(vaultPath, 'b.md'), '# Beta\n\nbeta content\n');

    // Index everything under the legacy 384-dim model.
    process.env.KG_EMBED_DIM = '384';
    const legacy = new Store(dbPath);
    await new IndexPipeline(legacy, fakeEmbedder(384)).index(vaultPath);
    expect(vecRowCount(legacy)).toBe(2);
    legacy.close();
    delete process.env.KG_EMBED_DIM;

    // A note is deleted while the app is offline, then the model changes.
    unlinkSync(join(vaultPath, 'b.md'));

    const store = new Store(dbPath);
    const stats = await new IndexPipeline(store, fakeEmbedder(768)).index(vaultPath);

    expect(vecTableSql(store)).toContain('float[768]');
    // a.md re-embedded even though its mtime never changed on disk.
    expect(stats.nodesIndexed).toBe(1);
    expect(vecRowCount(store)).toBe(1);
    // b.md fully reconciled — no ghost in nodes or full-text search.
    expect(store.getNode('b.md')).toBeUndefined();
    expect(store.searchFullText('beta')).toEqual([]);
    store.close();
  });
});
