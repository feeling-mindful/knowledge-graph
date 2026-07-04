import { homedir } from 'os';
import { join } from 'path';

export interface KGConfig {
  vaultPath: string;
  dataDir: string;
  dbPath: string;
}

export interface ConfigOverrides {
  vaultPath?: string;
  dataDir?: string;
}

function expandHome(p: string): string {
  return p.startsWith('~') ? join(homedir(), p.slice(1)) : p;
}

const DEFAULT_EMBED_DIM = 768;

/**
 * Embedding dimension for the sqlite-vec table. Must match the model chosen
 * via KG_EMBED_MODEL (bge-base-en-v1.5 → 768; MiniLM-L6 / bge-small → 384).
 * Invalid values fall back to the default rather than throwing at startup.
 */
export function resolveEmbedDim(): number {
  const raw = Number(process.env.KG_EMBED_DIM);
  return Number.isInteger(raw) && raw > 0 ? raw : DEFAULT_EMBED_DIM;
}

export function resolveConfig(overrides: ConfigOverrides): KGConfig {
  const vaultPath = overrides.vaultPath
    ?? process.env.KG_VAULT_PATH;

  if (!vaultPath) {
    throw new Error(
      'Vault path not configured. Set KG_VAULT_PATH or pass --vault-path.'
    );
  }

  const xdgData = process.env.XDG_DATA_HOME
    ?? join(homedir(), '.local', 'share');

  const dataDir = overrides.dataDir
    ?? process.env.KG_DATA_DIR
    ?? join(xdgData, 'knowledge-graph');

  return {
    vaultPath: expandHome(vaultPath),
    dataDir: expandHome(dataDir),
    dbPath: join(expandHome(dataDir), 'kg.db'),
  };
}
