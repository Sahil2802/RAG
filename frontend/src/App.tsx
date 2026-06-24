import { Header } from './components/Header';
import { ChatWindow } from './components/ChatWindow';
import { InputBar } from './components/InputBar';
import { useChat } from './hooks/useChat';

function ErrorBanner({ error, onDismiss }: { error: string; onDismiss: () => void }) {
  return (
    <div role="alert" className="px-4 py-2 bg-red-50 dark:bg-red-900/20 border-t border-red-200 dark:border-red-800">
      <div className="max-w-4xl mx-auto flex items-center justify-between gap-3">
        <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        <button
          onClick={onDismiss}
          aria-label="Dismiss error"
          className="cursor-pointer flex-shrink-0 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-200 focus:outline-none focus:ring-2 focus:ring-red-500 rounded p-0.5 transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
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
    <div className="flex flex-col h-screen bg-slate-50 dark:bg-slate-900">
      <Header />
      <ChatWindow messages={messages} isLoading={isLoading} />
      {error && <ErrorBanner error={error} onDismiss={clearError} />}
      <InputBar onSend={sendMessage} disabled={isLoading} />
    </div>
  );
}
