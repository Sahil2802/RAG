import { useRef, useState, KeyboardEvent } from 'react';

interface InputBarProps {
  onSend: (text: string) => void;
  disabled: boolean;
}

export function InputBar({ onSend, disabled }: InputBarProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (!value.trim() || disabled) return;
    onSend(value);
    setValue('');
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = !disabled && value.trim() !== '';

  return (
    <div
      className="sticky bottom-0 px-5 py-4"
      style={{
        background: 'var(--color-bg)',
        borderTop: '1px solid var(--color-border)',
      }}
    >
      <div
        className="flex items-end gap-3 max-w-3xl mx-auto rounded-xl p-1"
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
          outline: 'none',
          transition: 'border-color 0.15s ease',
        }}
        onFocusCapture={(e) => {
          (e.currentTarget as HTMLElement).style.borderColor = 'var(--color-amber)';
        }}
        onBlurCapture={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node)) {
            (e.currentTarget as HTMLElement).style.borderColor = 'var(--color-border)';
          }
        }}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents…"
          disabled={disabled}
          rows={1}
          aria-label="Chat input"
          className="flex-1 resize-none bg-transparent text-sm leading-relaxed px-3 py-2.5 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            color: 'var(--color-ink)',
            maxHeight: '8rem',
            overflowY: 'auto',
          }}
        />
        <button
          onClick={handleSend}
          disabled={!canSend}
          aria-label="Send message"
          className="flex items-center justify-center w-9 h-9 rounded-lg mb-0.5 mr-0.5 flex-shrink-0 transition-all duration-150 focus:outline-none disabled:opacity-30 disabled:cursor-not-allowed"
          style={{
            background: canSend ? 'var(--color-amber)' : 'var(--color-surface-2)',
            color: canSend ? 'oklch(0.12 0 0)' : 'var(--color-ink-subtle)',
            transform: 'scale(1)',
          }}
          onMouseEnter={(e) => {
            if (!canSend) return;
            (e.currentTarget as HTMLElement).style.background = 'oklch(0.70 0.145 44)';
          }}
          onMouseLeave={(e) => {
            if (!canSend) return;
            (e.currentTarget as HTMLElement).style.background = 'var(--color-amber)';
          }}
        >
          {disabled ? <SpinnerIcon /> : <SendIcon />}
        </button>
      </div>

      <p
        className="text-center text-[11px] mt-2"
        style={{ color: 'var(--color-ink-subtle)' }}
      >
        Enter to send · Shift+Enter for newline
      </p>
    </div>
  );
}

function SendIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 12h14M12 5l7 7-7 7" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="animate-spin"
    >
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}
