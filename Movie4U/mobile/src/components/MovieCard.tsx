import { Pressable, StyleSheet, Text } from 'react-native';
import type { Recommendation } from '../models/types';

export function MovieCard({ movie, onPress }: { movie: Recommendation; onPress: () => void }) {
  return <Pressable style={styles.card} onPress={onPress} accessibilityRole="button">
    <Text style={styles.title}>{movie.rank}. {movie.title}</Text>
    <Text style={styles.text}>{movie.genres.join(' · ')}</Text>
    <Text style={styles.text}>Ranking score: {movie.score.toFixed(3)}</Text>
    <Text style={styles.hint}>View details and KG evidence →</Text>
  </Pressable>;
}
const styles = StyleSheet.create({
  card: { padding: 16, borderRadius: 12, backgroundColor: '#172033', marginBottom: 10 },
  title: { color: '#f8fafc', fontSize: 17, fontWeight: 'bold' },
  text: { color: '#aeb8c8', marginTop: 6 },
  hint: { color: '#7dd3fc', marginTop: 10 },
});
