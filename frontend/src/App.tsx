import { Header } from './components/Header';
import { ChatWindow } from './components/ChatWindow';
import { InputBar } from './components/InputBar';
import { useChat } from './hooks/useChat';

function ErrorBanner({ error, onDismiss }: { error: string; onDismiss: () => void }) {
  return (
    <div
      role="alert"
      className="px-6 py-2.5"
      style={{
        background: 'var(--color-error-surface)',
        borderTop: '1px solid oklch(0.30 0.08 25)',
      }}
    >
      <div className="max-w-3xl mx-auto flex items-center justify-between gap-3">
        <p className="text-sm" style={{ color: 'var(--color-error)' }}>
          {error}
        </p>
        <button
          onClick={onDismiss}
          aria-label="Dismiss error"
          className="flex-shrink-0 cursor-pointer rounded p-0.5 focus:outline-none transition-opacity hover:opacity-70"
          style={{ color: 'var(--color-error)' }}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const { messages, isLoading, error, sendMessage, clearError } = useChat();

  return (
    <div
      className="flex flex-col h-screen"
      style={{ background: 'var(--color-bg)' }}
    >
      <Header />
      <ChatWindow messages={messages} isLoading={isLoading} />
      {error && <ErrorBanner error={error} onDismiss={clearError} />}
      <InputBar onSend={sendMessage} disabled={isLoading} />
    </div>
  );
}
