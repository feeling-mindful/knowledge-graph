import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import type { ParsedNode, ParsedEdge, SearchResult } from './types.js';
import { resolveEmbedDim } from './config.js';

export class Store {
  db: Database.Database;
  /** Dimension of the nodes_vec table as it exists on disk. */
  private vecDim: number;

  constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.db.pragma('busy_timeout = 5000');
    sqliteVec.load(this.db);
    this.createSchema();
    this.vecDim = this.readVecDim() ?? resolveEmbedDim();
  }

  private createSchema(): void {
    const dim = resolveEmbedDim();
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT,
        frontmatter TEXT NOT NULL DEFAULT '{}'
      );

      CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        context TEXT NOT NULL DEFAULT ''
      );

      CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
      CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

      CREATE TABLE IF NOT EXISTS communities (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        node_ids TEXT NOT NULL DEFAULT '[]'
      );

      CREATE TABLE IF NOT EXISTS sync (
        path TEXT PRIMARY KEY,
        mtime INTEGER NOT NULL,
        indexed_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS centrality (
        node_id TEXT PRIMARY KEY,
        pagerank REAL NOT NULL DEFAULT 0
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
        USING fts5(title, content, content='nodes', content_rowid='rowid');

      CREATE VIRTUAL TABLE IF NOT EXISTS nodes_vec
        USING vec0(embedding float[${dim}]);
    `);
  }

  /** Parse the on-disk nodes_vec dimension out of its DDL in sqlite_master. */
  private readVecDim(): number | undefined {
    const row = this.db.prepare(
      "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'nodes_vec'"
    ).get() as { sql: string } | undefined;
    if (!row?.sql) return undefined;
    const match = /float\[(\d+)\]/i.exec(row.sql);
    return match ? Number(match[1]) : undefined;
  }

  /**
   * Rebuild nodes_vec when the embedder's actual output dimension no longer
   * matches the table (e.g. a DB indexed with 384-dim MiniLM after the switch
   * to 768-dim bge-base). CREATE ... IF NOT EXISTS keeps the old dimension
   * forever, and every insert/search then fails with a dimension-mismatch
   * error. Keyed off real embedding lengths — never config alone — so a
   * mistyped KG_EMBED_DIM can never destroy a valid index.
   *
   * Embeddings are derivable from vault content, so the vec rows are
   * droppable; sync mtimes are zeroed (not deleted — the pipeline's
   * deleted-file reconciliation diffs sync paths against disk, and losing
   * them would permanently ghost notes removed before the rebuild) so the
   * next index pass re-embeds every file. Single transaction: a crash
   * mid-rebuild must not leave an empty vec table with live sync mtimes.
   */
  ensureVecDim(dim: number): void {
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new Error(`Invalid embedding dimension: ${dim}`);
    }
    if (dim === this.vecDim) return;
    const previousDim = this.vecDim;
    this.db.transaction(() => {
      this.db.exec('DROP TABLE IF EXISTS nodes_vec');
      this.db.exec(`CREATE VIRTUAL TABLE nodes_vec USING vec0(embedding float[${dim}])`);
      this.db.exec('UPDATE sync SET mtime = 0');
    })();
    this.vecDim = dim;
    console.error(
      `knowledge-graph: nodes_vec dimension changed (${previousDim} → ${dim}); ` +
      'rebuilt vector index — every note re-embeds on the next index pass'
    );
  }

  upsertNode(node: ParsedNode): void {
    // FTS5 content-sync tables require manual delete-before-reinsert.
    // We must fetch the ACTUAL old values for the FTS5 delete command.
    const existing = this.db.prepare(
      'SELECT rowid, title, content FROM nodes WHERE id = ?'
    ).get(node.id) as { rowid: number; title: string; content: string } | undefined;

    if (existing) {
      this.db.prepare(
        "INSERT INTO nodes_fts(nodes_fts, rowid, title, content) VALUES('delete', ?, ?, ?)"
      ).run(existing.rowid, existing.title, existing.content);
    }

    this.db.prepare(`
      INSERT INTO nodes (id, title, content, frontmatter)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        title = excluded.title,
        content = excluded.content,
        frontmatter = excluded.frontmatter
    `).run(node.id, node.title, node.content, JSON.stringify(node.frontmatter));

    const row = this.db.prepare(
      'SELECT rowid FROM nodes WHERE id = ?'
    ).get(node.id) as { rowid: number };

    this.db.prepare(
      'INSERT INTO nodes_fts(rowid, title, content) VALUES(?, ?, ?)'
    ).run(row.rowid, node.title, node.content);
  }

  getNode(id: string): (ParsedNode & { rowid: number }) | undefined {
    const row = this.db.prepare(
      'SELECT rowid, id, title, content, frontmatter FROM nodes WHERE id = ?'
    ).get(id) as any;
    if (!row) return undefined;
    return {
      id: row.id,
      title: row.title,
      content: row.content,
      frontmatter: JSON.parse(row.frontmatter),
      rowid: row.rowid,
    };
  }

  allNodeIds(): string[] {
    return this.db.prepare('SELECT id FROM nodes').all().map((r: any) => r.id);
  }

  insertEdge(edge: ParsedEdge): void {
    this.db.prepare(
      'INSERT INTO edges (source_id, target_id, context) VALUES (?, ?, ?)'
    ).run(edge.sourceId, edge.targetId, edge.context);
  }

  getEdgesFrom(nodeId: string): Array<ParsedEdge & { id: number }> {
    return this.db.prepare(
      'SELECT id, source_id, target_id, context FROM edges WHERE source_id = ?'
    ).all(nodeId).map((r: any) => ({
      id: r.id,
      sourceId: r.source_id,
      targetId: r.target_id,
      context: r.context,
    }));
  }

  getEdgesTo(nodeId: string): Array<ParsedEdge & { id: number }> {
    return this.db.prepare(
      'SELECT id, source_id, target_id, context FROM edges WHERE target_id = ?'
    ).all(nodeId).map((r: any) => ({
      id: r.id,
      sourceId: r.source_id,
      targetId: r.target_id,
      context: r.context,
    }));
  }

  countEdgesFrom(nodeId: string): number {
    const row = this.db.prepare(
      'SELECT COUNT(*) as cnt FROM edges WHERE source_id = ?'
    ).get(nodeId) as { cnt: number };
    return row.cnt;
  }

  countEdgesTo(nodeId: string): number {
    const row = this.db.prepare(
      'SELECT COUNT(*) as cnt FROM edges WHERE target_id = ?'
    ).get(nodeId) as { cnt: number };
    return row.cnt;
  }

  getEdgeSummariesFrom(nodeId: string): Array<{ nodeId: string; title: string }> {
    return this.db.prepare(`
      SELECT DISTINCT e.target_id, n.title
      FROM edges e
      LEFT JOIN nodes n ON n.id = e.target_id
      WHERE e.source_id = ?
    `).all(nodeId).map((r: any) => ({
      nodeId: r.target_id,
      title: r.title ?? r.target_id,
    }));
  }

  getEdgeSummariesTo(nodeId: string): Array<{ nodeId: string; title: string }> {
    return this.db.prepare(`
      SELECT DISTINCT e.source_id, n.title
      FROM edges e
      LEFT JOIN nodes n ON n.id = e.source_id
      WHERE e.target_id = ?
    `).all(nodeId).map((r: any) => ({
      nodeId: r.source_id,
      title: r.title ?? r.source_id,
    }));
  }

  deleteNode(id: string): void {
    // FTS5 delete requires actual old values, not empty strings
    const row = this.db.prepare(
      'SELECT rowid, title, content FROM nodes WHERE id = ?'
    ).get(id) as { rowid: number; title: string; content: string } | undefined;

    if (row) {
      this.db.prepare(
        "INSERT INTO nodes_fts(nodes_fts, rowid, title, content) VALUES('delete', ?, ?, ?)"
      ).run(row.rowid, row.title, row.content);
      // sqlite-vec requires BigInt rowids via better-sqlite3
      this.db.prepare('DELETE FROM nodes_vec WHERE rowid = ?').run(BigInt(row.rowid));
    }

    this.db.prepare('DELETE FROM nodes WHERE id = ?').run(id);
    this.db.prepare('DELETE FROM edges WHERE source_id = ? OR target_id = ?').run(id, id);
    this.db.prepare('DELETE FROM sync WHERE path = ?').run(id);
    this.db.prepare('DELETE FROM centrality WHERE node_id = ?').run(id);
  }

  deleteAllEdgesFrom(nodeId: string): void {
    this.db.prepare('DELETE FROM edges WHERE source_id = ?').run(nodeId);
  }

  searchFullText(query: string, limit = 20): SearchResult[] {
    return this.db.prepare(`
      SELECT n.id, n.title, rank,
        snippet(nodes_fts, 1, '>>>', '<<<', '...', 40) as excerpt
      FROM nodes_fts f
      JOIN nodes n ON n.rowid = f.rowid
      WHERE nodes_fts MATCH ?
      ORDER BY rank
      LIMIT ?
    `).all(query, clampLimit(limit)).map((r: any) => ({
      nodeId: r.id,
      title: r.title,
      score: -r.rank,
      excerpt: r.excerpt ?? '',
    }));
  }

  upsertEmbedding(nodeId: string, embedding: Float32Array): void {
    this.ensureVecDim(embedding.length);
    const node = this.getNode(nodeId);
    if (!node) return;
    // sqlite-vec requires BigInt rowids via better-sqlite3
    this.db.prepare('DELETE FROM nodes_vec WHERE rowid = ?').run(BigInt(node.rowid));
    this.db.prepare(
      'INSERT INTO nodes_vec(rowid, embedding) VALUES (?, ?)'
    ).run(BigInt(node.rowid), Buffer.from(embedding.buffer));
  }

  searchVector(embedding: Float32Array, limit = 20): SearchResult[] {
    // A query at the wrong dimension means the index predates a model change
    // and has not been re-indexed yet. Degrade to no dense results (a MATCH
    // would throw) — kg_index rebuilds the table via the embedding write path.
    if (embedding.length !== this.vecDim) {
      console.error(
        `knowledge-graph: query embedding is ${embedding.length}-dim but the vector index is ` +
        `${this.vecDim}-dim — run kg_index to rebuild before semantic search returns results`
      );
      return [];
    }
    return this.db.prepare(`
      SELECT v.rowid, v.distance, n.id, n.title, n.content
      FROM nodes_vec v
      JOIN nodes n ON n.rowid = v.rowid
      WHERE embedding MATCH ? AND k = ?
      ORDER BY distance
    `).all(Buffer.from(embedding.buffer), clampLimit(limit)).map((r: any) => ({
      nodeId: r.id,
      title: r.title,
      score: 1 - r.distance,
      excerpt: firstParagraph(r.content ?? '', 200),
    }));
  }

  upsertSync(path: string, mtime: number): void {
    this.db.prepare(`
      INSERT INTO sync (path, mtime, indexed_at) VALUES (?, ?, ?)
      ON CONFLICT(path) DO UPDATE SET mtime = excluded.mtime, indexed_at = excluded.indexed_at
    `).run(path, mtime, Date.now());
  }

  getSyncMtime(path: string): number | undefined {
    const row = this.db.prepare(
      'SELECT mtime FROM sync WHERE path = ?'
    ).get(path) as { mtime: number } | undefined;
    return row?.mtime;
  }

  getAllSyncPaths(): Set<string> {
    return new Set(
      this.db.prepare('SELECT path FROM sync').all().map((r: any) => r.path)
    );
  }

  upsertCommunity(community: { id: number; label: string; summary: string; nodeIds: string[] }): void {
    this.db.prepare(`
      INSERT INTO communities (id, label, summary, node_ids) VALUES (?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        label = excluded.label,
        summary = excluded.summary,
        node_ids = excluded.node_ids
    `).run(community.id, community.label, community.summary, JSON.stringify(community.nodeIds));
  }

  clearCommunities(): void {
    this.db.prepare('DELETE FROM communities').run();
  }

  getAllCommunities(): Array<{ id: number; label: string; summary: string; nodeIds: string[] }> {
    return this.db.prepare('SELECT * FROM communities').all().map((r: any) => ({
      id: r.id,
      label: r.label,
      summary: r.summary,
      nodeIds: JSON.parse(r.node_ids),
    }));
  }

  // ─── Centrality (normalized PageRank in [0,1], persisted at index time) ───

  upsertCentrality(nodeId: string, pagerank: number): void {
    this.db.prepare(`
      INSERT INTO centrality (node_id, pagerank) VALUES (?, ?)
      ON CONFLICT(node_id) DO UPDATE SET pagerank = excluded.pagerank
    `).run(nodeId, pagerank);
  }

  getCentrality(nodeId: string): number {
    const row = this.db.prepare(
      'SELECT pagerank FROM centrality WHERE node_id = ?'
    ).get(nodeId) as { pagerank: number } | undefined;
    return row?.pagerank ?? 0;
  }

  /** Batch-fetch centrality for a set of node ids (search re-ranking). */
  getCentralityMap(nodeIds: string[]): Map<string, number> {
    const map = new Map<string, number>();
    if (nodeIds.length === 0) return map;
    const placeholders = nodeIds.map(() => '?').join(',');
    const rows = this.db.prepare(
      `SELECT node_id, pagerank FROM centrality WHERE node_id IN (${placeholders})`
    ).all(...nodeIds) as Array<{ node_id: string; pagerank: number }>;
    for (const r of rows) map.set(r.node_id, r.pagerank);
    return map;
  }

  clearCentrality(): void {
    this.db.prepare('DELETE FROM centrality').run();
  }

  close(): void {
    this.db.close();
  }
}

/**
 * Clamp a caller-supplied limit at the storage boundary: MCP clients and the
 * CLI can hand us NaN (parseInt of garbage), zero, or negatives, none of which
 * belong in a SQL LIMIT or a vec0 k.
 */
function clampLimit(limit: number, fallback = 20): number {
  return Number.isFinite(limit) && limit >= 1 ? Math.floor(limit) : fallback;
}

function firstParagraph(content: string, maxLen: number): string {
  const para = content.split(/\n\n+/).find(p => p.trim().length > 0 && !p.startsWith('#'));
  if (!para) return '';
  const trimmed = para.trim();
  return trimmed.length > maxLen ? trimmed.slice(0, maxLen) + '...' : trimmed;
}
