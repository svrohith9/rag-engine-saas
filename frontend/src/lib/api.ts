export type CreateSessionResponse = { session_id: string };
export type FileInfo = {
  id: string;
  name: string;
  mime: string;
  size_bytes: number;
  created_at: string;
};

export type UploadResult = {
  ok: boolean;
  reason?: string;
  file_id?: string;
  stored_as?: string;
  mime?: string;
  size_bytes?: number;
  added_chunks?: number;
  added_embeddings?: number;
};

export type UploadResponse = { session_id: string; results: UploadResult[] };

export type Citation = {
  chunk_id: string;
  file_name: string;
  page?: number | null;
  score: number;
  snippet: string;
};

export type ChatResponse = {
  answer: string;
  citations: Citation[];
  used_embeddings: boolean;
  model: string;
};

export function getBackendUrl() {
  return localStorage.getItem('backendUrl') || 'http://localhost:8000';
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function createSession(): Promise<CreateSessionResponse> {
  return jsonFetch(`${getBackendUrl()}/api/sessions`, { method: 'POST' });
}

export async function listFiles(sessionId: string): Promise<FileInfo[]> {
  return jsonFetch(`${getBackendUrl()}/api/sessions/${sessionId}/files`);
}

export async function uploadFiles(sessionId: string, files: File[]): Promise<UploadResponse> {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  return jsonFetch(`${getBackendUrl()}/api/sessions/${sessionId}/files`, { method: 'POST', body: fd });
}

export async function chat(sessionId: string, message: string): Promise<ChatResponse> {
  return jsonFetch(`${getBackendUrl()}/api/sessions/${sessionId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, top_k: 8, use_images: true }),
  });
}
