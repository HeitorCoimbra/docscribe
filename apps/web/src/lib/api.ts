const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

async function apiFetch<T>(
  path: string,
  token: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_URL}/api/v1${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getThreads(token: string) {
  return apiFetch<import('@/types/session').ThreadGroup[]>('/threads', token);
}

export async function createThread(token: string) {
  return apiFetch<import('@/types/session').Thread>('/threads', token, {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function getThread(token: string, id: string) {
  return apiFetch<import('@/types/session').ThreadDetail>(`/threads/${id}`, token);
}

export async function deleteThread(token: string, id: string) {
  return apiFetch<void>(`/threads/${id}`, token, { method: 'DELETE' });
}

export async function getMessages(token: string, threadId: string) {
  return apiFetch<import('@/types/session').Message[]>(
    `/threads/${threadId}/messages`,
    token
  );
}

export function getApiUrl() {
  return API_URL;
}
