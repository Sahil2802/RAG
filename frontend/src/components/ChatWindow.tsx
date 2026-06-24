import { useEffect, useRef } from 'react';
import type { Message as MessageType } from '../types';
import { Message } from './Message';

interface ChatWindowProps {
  messages: MessageType[];
  isLoading: boolean;
}

export function ChatWindow({ messages }: ChatWindowProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const { scrollTop, scrollHeight, clientHeight } = container;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 120;
    if (isNearBottom) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-10">
        <div className="text-center max-w-xs">
          <p
            className="text-2xl font-semibold mb-2"
            style={{ color: 'var(--color-ink)', letterSpacing: '-0.02em' }}
          >
            Ask your documents.
          </p>
          <p
            className="text-sm leading-relaxed"
            style={{ color: 'var(--color-ink-subtle)' }}
          >
            Your knowledge base is indexed and ready. Questions, summaries, comparisons — anything in the text.
          </p>

          <div
            className="mt-8 flex flex-col gap-2 text-left"
            aria-label="Example questions"
          >
            {EXAMPLES.map((q) => (
              <div
                key={q}
                className="px-3 py-2 rounded-md text-xs"
                style={{
                  background: 'var(--color-surface)',
                  color: 'var(--color-ink-muted)',
                  border: '1px solid var(--color-border)',
                }}
              >
                {q}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto px-5 py-7">
      <div className="max-w-3xl mx-auto flex flex-col gap-6">
        {messages.map((msg, i) => (
          <Message key={msg.id} message={msg} index={i} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

const EXAMPLES = [
  'Summarize the key findings in the uploaded paper.',
  'What methodology was used and what are its limitations?',
  'Compare the conclusions across all documents.',
];
