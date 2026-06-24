export interface Chunk {
  content: string;
  source: string;
  similarity_score: number;
  rank: number;
}

export interface Sources {
  citations: string[];
  confidence: number;
  chunks: Chunk[];
}

export type Role = 'user' | 'assistant';

export interface Message {
  id: string;
  role: Role;
  content: string;
  sources?: Sources;
  isStreaming?: boolean;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}
