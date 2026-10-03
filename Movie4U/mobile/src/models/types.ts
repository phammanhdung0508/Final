export type Method = 'kg_only' | 'kg_gnn';
export type Evidence = { liked_movie_id: number; liked_title: string; genre: string; path: string };
export type Movie = { movie_id: number; title: string; genres: string[] };
export type Recommendation = Movie & { rank: number; score: number; evidence: Evidence[] };
export type RecommendationsResponse = { user_id: number; method: Method; score_note: string; recommendations: Recommendation[] };
export type User = { user_id: number; training_ratings: number };
export type Evaluation = { models: Record<Method, { test: Record<string, number> }> };
