import { useState, useCallback } from 'react';
import type { Message, Sources } from '../types';

export function useChat(): {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (input: string) => Promise<void>;
  clearError: () => void;
} {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async (input: string): Promise<void> => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userId = crypto.randomUUID();
    const assistantId = crypto.randomUUID();

    const userMessage: Message = {
      id: userId,
      role: 'user',
      content: trimmed,
    };

    const assistantPlaceholder: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      isStreaming: true,
    };

    // Capture the messages before we append the new user message
    // so we can send existing conversation history to the backend
    let existingMessages: Message[] = [];
    setMessages((prev) => {
      existingMessages = prev;
      return [...prev, userMessage, assistantPlaceholder];
    });

    setIsLoading(true);
    setError(null);

    // Build the body: existing messages + new user message, stripped to role + content only
    const bodyMessages = [...existingMessages, userMessage].map(({ role, content }) => ({
      role,
      content,
    }));

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: bodyMessages }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Split on double newline — each SSE event block ends with \n\n
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          if (!part.trim()) continue;

          // Parse event name and data from the block
          let eventName = '';
          let dataLine = '';

          for (const line of part.split('\n')) {
            if (line.startsWith('event: ')) {
              eventName = line.slice('event: '.length).trim();
            } else if (line.startsWith('data: ')) {
              dataLine = line.slice('data: '.length).trim();
            }
          }

          if (!eventName || !dataLine) continue;

          let data: Record<string, unknown>;
          try {
            data = JSON.parse(dataLine);
          } catch {
            continue;
          }

          if (eventName === 'sources') {
            const sources = data as unknown as Sources;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, sources } : msg
              )
            );
          } else if (eventName === 'token') {
            const text = typeof data.text === 'string' ? data.text : '';
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? { ...msg, content: msg.content + text }
                  : msg
              )
            );
          } else if (eventName === 'done') {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, isStreaming: false } : msg
              )
            );
            setIsLoading(false);
          } else if (eventName === 'error') {
            const message =
              typeof data.message === 'string' ? data.message : 'Unknown error';
            setError(message);
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, isStreaming: false } : msg
              )
            );
            setIsLoading(false);
          }
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred';
      setError(message);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId ? { ...msg, isStreaming: false } : msg
        )
      );
      setIsLoading(false);
    }
  }, [isLoading]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return { messages, isLoading, error, sendMessage, clearError };
}
