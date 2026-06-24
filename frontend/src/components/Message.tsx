import type { Message as MessageType } from '../types';
import { SourcesPanel } from './SourcesPanel';

interface MessageProps {
  message: MessageType;
  index: number;
}

export function Message({ message, index }: MessageProps) {
  const delay = Math.min(index * 20, 80);

  if (message.role === 'user') {
    return (
      <div
        className="flex justify-end message-enter"
        style={{ animationDelay: `${delay}ms` }}
      >
        <div
          className="max-w-[72%] px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm leading-relaxed whitespace-pre-wrap break-words"
          style={{
            background: 'var(--color-amber-dim)',
            color: 'var(--color-ink)',
            border: '1px solid oklch(0.30 0.06 48)',
          }}
        >
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex justify-start message-enter"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="w-full flex flex-col gap-2">
        {message.sources && <SourcesPanel sources={message.sources} />}

        <div
          className="text-sm leading-[1.75] whitespace-pre-wrap break-words"
          style={{ color: 'var(--color-ink)' }}
        >
          {message.isStreaming && message.content === '' ? (
            <div className="flex items-center gap-1.5 h-5">
              <span
                className="typing-dot w-1.5 h-1.5 rounded-full inline-block"
                style={{ background: 'var(--color-ink-subtle)' }}
              />
              <span
                className="typing-dot w-1.5 h-1.5 rounded-full inline-block"
                style={{ background: 'var(--color-ink-subtle)' }}
              />
              <span
                className="typing-dot w-1.5 h-1.5 rounded-full inline-block"
                style={{ background: 'var(--color-ink-subtle)' }}
              />
            </div>
          ) : (
            <p className={message.isStreaming ? 'streaming-cursor' : ''}>
              {message.content}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
