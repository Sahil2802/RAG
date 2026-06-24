import { Message as MessageType } from '../types';
import { SourcesPanel } from './SourcesPanel';

interface MessageProps {
  message: MessageType;
}

export function Message({ message }: MessageProps) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap break-words shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
        {message.sources && <SourcesPanel sources={message.sources} />}
        {message.isStreaming && message.content === '' ? (
          <div className="flex items-center gap-1 h-5">
            <span className="typing-dot w-2 h-2 rounded-full bg-slate-400 dark:bg-slate-500 inline-block" />
            <span className="typing-dot w-2 h-2 rounded-full bg-slate-400 dark:bg-slate-500 inline-block" />
            <span className="typing-dot w-2 h-2 rounded-full bg-slate-400 dark:bg-slate-500 inline-block" />
          </div>
        ) : (
          <p className={`text-sm leading-relaxed text-slate-800 dark:text-slate-200 whitespace-pre-wrap break-words${message.isStreaming ? ' streaming-cursor' : ''}`}>
            {message.content}
          </p>
        )}
      </div>
    </div>
  );
}
