import { useState } from 'react';
import type { Sources } from '../types';

interface SourcesPanelProps {
  sources: Sources;
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const basename = (path: string) => path.split(/[\\/]/).pop() ?? path;
  const pct = Math.round(sources.confidence * 100);

  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        className="flex items-center gap-2 text-xs transition-colors focus:outline-none rounded-sm"
        style={{
          color: isOpen ? 'var(--color-amber)' : 'var(--color-ink-subtle)',
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.color = 'var(--color-amber)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.color = isOpen
            ? 'var(--color-amber)'
            : 'var(--color-ink-subtle)';
        }}
      >
        <span
          className="inline-flex items-center justify-center w-4 h-4 rounded text-[10px] font-semibold leading-none"
          style={{
            background: isOpen ? 'var(--color-amber-dim)' : 'var(--color-surface-2)',
            color: isOpen ? 'var(--color-amber)' : 'var(--color-ink-subtle)',
            border: `1px solid ${isOpen ? 'oklch(0.30 0.06 48)' : 'var(--color-border)'}`,
          }}
        >
          {sources.chunks.length}
        </span>
        <span>
          {sources.chunks.length === 1 ? 'reference' : 'references'}
        </span>
        <span style={{ color: 'var(--color-ink-subtle)' }}>·</span>
        <span>{pct}% confidence</span>
        <ChevronIcon isOpen={isOpen} />
      </button>

      <div
        className="overflow-hidden transition-all duration-200 ease-out"
        style={{ maxHeight: isOpen ? '700px' : '0px', opacity: isOpen ? 1 : 0 }}
      >
        <div className="mt-2 flex flex-col gap-1.5">
          {sources.chunks.map((chunk, i) => (
            <div
              key={i}
              className={`rounded-lg p-3 ${isOpen ? 'source-enter' : ''}`}
              style={{
                background: 'var(--color-surface-2)',
                border: '1px solid var(--color-border)',
                animationDelay: `${i * 40}ms`,
              }}
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <span
                  className="text-xs font-medium truncate"
                  style={{ color: 'var(--color-ink-muted)' }}
                >
                  {basename(chunk.source)}
                </span>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span
                    className="text-xs tabular-nums"
                    style={{ color: 'var(--color-ink-subtle)' }}
                  >
                    {Math.round(chunk.similarity_score * 100)}%
                  </span>
                  <span
                    className="inline-flex items-center justify-center w-5 h-5 rounded text-[10px] font-semibold"
                    style={{
                      background: 'var(--color-amber-dim)',
                      color: 'var(--color-amber)',
                    }}
                  >
                    {chunk.rank}
                  </span>
                </div>
              </div>
              <p
                className="text-xs leading-relaxed"
                style={{ color: 'var(--color-ink-subtle)' }}
              >
                {chunk.content.length > 180
                  ? chunk.content.slice(0, 180) + '…'
                  : chunk.content}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ChevronIcon({ isOpen }: { isOpen: boolean }) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="transition-transform duration-200"
      style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}
