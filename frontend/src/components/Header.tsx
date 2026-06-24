export function Header() {
  return (
    <header
      className="sticky top-0 z-40 border-b"
      style={{
        background: 'var(--color-surface)',
        borderColor: 'var(--color-border)',
      }}
    >
      <div className="flex items-center justify-between px-6 h-14">
        <div className="flex items-center gap-3">
          <ArchiveIcon />
          <div>
            <span
              className="text-sm font-semibold tracking-tight"
              style={{ color: 'var(--color-ink)' }}
            >
              RAG Chat
            </span>
            <span
              className="text-xs ml-2 font-normal"
              style={{ color: 'var(--color-ink-subtle)' }}
            >
              Llama 3.2 · FAISS
            </span>
          </div>
        </div>

        <div
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium"
          style={{
            background: 'var(--color-amber-dim)',
            color: 'var(--color-amber)',
          }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80" />
          local
        </div>
      </div>
    </header>
  );
}

function ArchiveIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{ color: 'var(--color-amber)', flexShrink: 0 }}
    >
      <path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z" />
      <path d="m3.3 7 8.7 5 8.7-5" />
      <path d="M12 22V12" />
    </svg>
  );
}
