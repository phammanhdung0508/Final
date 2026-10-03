import type { Method, Movie, User, Evaluation, RecommendationsResponse } from '../models/types';

declare const process: { env: { EXPO_PUBLIC_API_URL?: string } };
export const API_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';

async function get<T>(path: string): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`${API_URL}${path}`, { signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail ?? `HTTP ${response.status}`);
    return data as T;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  users: () => get<User[]>('/users'),
  recommend: (user: number, method: Method) => get<RecommendationsResponse>(`/users/${user}/recommendations?method=${method}&k=10`),
  movie: (movie: number) => get<Movie>(`/movies/${movie}`),
  evaluation: () => get<Evaluation>('/evaluation'),
};
