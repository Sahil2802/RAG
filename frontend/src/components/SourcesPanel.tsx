import { useState } from 'react';
import { Sources } from '../types';

interface SourcesPanelProps {
  sources: Sources;
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const basename = (path: string) => path.split(/[\\/]/).pop() ?? path;
  const pct = Math.round(sources.confidence * 100);

  return (
    <div className="mb-3">
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        className="cursor-pointer flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-md px-1 py-0.5 transition-colors"
      >
        <DocumentIcon />
        <span>{sources.chunks.length} source{sources.chunks.length !== 1 ? 's' : ''}</span>
        <span className="text-slate-400">·</span>
        <span>{pct}% confidence</span>
        <ChevronIcon isOpen={isOpen} />
      </button>

      <div
        className="overflow-hidden transition-all duration-200 ease-out"
        style={{ maxHeight: isOpen ? '600px' : '0px' }}
      >
        <div className="mt-2 space-y-2">
          {sources.chunks.map((chunk, i) => (
            <div
              key={i}
              className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3 border border-slate-200 dark:border-slate-600"
            >
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <span className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate">
                  {basename(chunk.source)}
                </span>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    {Math.round(chunk.similarity_score * 100)}%
                  </span>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300">
                    #{chunk.rank}
                  </span>
                </div>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                {chunk.content.length > 150 ? chunk.content.slice(0, 150) + '…' : chunk.content}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DocumentIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}

function ChevronIcon({ isOpen }: { isOpen: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={`transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}
