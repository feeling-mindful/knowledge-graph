import { describe, it, expect } from 'vitest';
import { rrfFuse, toFtsQuery } from '../src/lib/fusion.js';

describe('rrfFuse', () => {
  it('ranks ids present in both lists above single-list ids', () => {
    const fused = rrfFuse([
      ['both', 'dense-only'],
      ['both', 'lexical-only'],
    ]);
    expect(fused.get('both')!).toBeGreaterThan(fused.get('dense-only')!);
    expect(fused.get('both')!).toBeGreaterThan(fused.get('lexical-only')!);
  });

  it('rewards higher rank within a list', () => {
    const fused = rrfFuse([['first', 'second', 'third']]);
    expect(fused.get('first')!).toBeGreaterThan(fused.get('second')!);
    expect(fused.get('second')!).toBeGreaterThan(fused.get('third')!);
  });

  it('uses 1-based rank with the k constant', () => {
    const fused = rrfFuse([['a']], 60);
    expect(fused.get('a')).toBeCloseTo(1 / 61, 10);
  });

  it('returns empty map for empty input', () => {
    expect(rrfFuse([]).size).toBe(0);
    expect(rrfFuse([[], []]).size).toBe(0);
  });
});

describe('toFtsQuery', () => {
  it('quotes tokens so FTS5 operators and punctuation are inert', () => {
    expect(toFtsQuery("what's this? (thing) OR that*")).toBe(
      '"what" OR "s" OR "this" OR "thing" OR "OR" OR "that"'
    );
  });

  it('handles plain multi-word queries', () => {
    expect(toFtsQuery('graph theory')).toBe('"graph" OR "theory"');
  });

  it('keeps unicode letters and digits', () => {
    expect(toFtsQuery('café 42')).toBe('"café" OR "42"');
  });

  it('returns empty string when no tokens survive', () => {
    expect(toFtsQuery('?!* ()')).toBe('');
    expect(toFtsQuery('')).toBe('');
  });
});
