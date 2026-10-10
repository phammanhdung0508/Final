import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Modal, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { api, API_URL } from '../services/api';
import { MovieCard } from '../components/MovieCard';
import type { Evaluation, Recommendation } from '../models/types';

export function HomeScreen() {
  const [user, setUser] = useState('1');
  const [usersCount, setUsersCount] = useState<number>();
  const [results, setResults] = useState<{ kg_only: Recommendation[]; kg_gnn: Recommendation[] }>();
  const [metrics, setMetrics] = useState<Evaluation>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<Recommendation>();
  const request = useRef(0);

  useEffect(() => {
    let active = true;
    api.users().then(rows => { if (active) setUsersCount(rows.length); }).catch(e => { if (active) setError(String(e)); });
    api.evaluation().then(rows => { if (active) setMetrics(rows); }).catch(() => {});
    return () => { active = false; request.current++; };
  }, []);

  async function compare() {
    if (!/^\d+$/.test(user) || Number(user) < 1) { setError('Enter a valid MovieLens user ID.'); return; }
    const id = ++request.current;
    setBusy(true); setError(''); setResults(undefined);
    try {
      const [kg, gnn] = await Promise.all([api.recommend(Number(user), 'kg_only'), api.recommend(Number(user), 'kg_gnn')]);
      if (request.current === id) setResults({ kg_only: kg.recommendations, kg_gnn: gnn.recommendations });
    } catch (e) {
      if (request.current === id) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (request.current === id) setBusy(false);
    }
  }

  return <View style={styles.root}><ScrollView contentContainerStyle={styles.content}>
    <Text style={styles.heading}>Movie4U</Text>
    <Text style={styles.text}>One Knowledge Graph. Two recommendation approaches.</Text>
    <Text style={styles.text}>Demo profile: existing anonymized MovieLens user{usersCount ? ` (${usersCount} available)` : ''}.</Text>
    <TextInput style={styles.input} value={user} onChangeText={setUser} keyboardType="number-pad" placeholder="User ID" placeholderTextColor="#94a3b8" accessibilityLabel="MovieLens user ID" />
    <Pressable style={styles.button} onPress={compare} disabled={busy} accessibilityRole="button"><Text style={styles.buttonText}>Compare recommendations</Text></Pressable>
    {busy && <ActivityIndicator color="#7dd3fc" style={{ margin: 16 }} />}
    {!!error && <Text style={styles.error}>{error}</Text>}
    {(['kg_only', 'kg_gnn'] as const).map(method => <View key={method}>
      <Text style={styles.subheading}>{method === 'kg_only' ? 'KG-only' : 'KG + GraphSAGE'}</Text>
      {metrics && <Text style={styles.text}>Test Recall@10: {metrics.models[method].test['recall@10'].toFixed(4)} · NDCG@10: {metrics.models[method].test['ndcg@10'].toFixed(4)}</Text>}
      {results?.[method].map(movie => <MovieCard key={movie.movie_id} movie={movie} onPress={() => setSelected(movie)} />)}
    </View>)}
    <Text style={styles.text}>Scores are not probabilities and cannot be compared across methods. KG evidence is not a causal explanation of GNN scores.</Text>
    <Text style={styles.text}>Backend: {API_URL}</Text>
  </ScrollView>
  <Modal visible={!!selected} onRequestClose={() => setSelected(undefined)} animationType="slide">
    <ScrollView contentContainerStyle={[styles.content, { paddingTop: 60 }]} style={styles.root}>
      <Pressable onPress={() => setSelected(undefined)} accessibilityRole="button"><Text style={styles.link}>← Back</Text></Pressable>
      <Text style={styles.heading}>{selected?.title}</Text>
      <Text style={styles.text}>{selected?.genres.join(' · ')}</Text>
      <Text style={styles.subheading}>Training-history KG evidence</Text>
      {selected?.evidence.length ? selected.evidence.map((e, i) => <Text style={styles.evidence} key={i}>{e.path}</Text>) : <Text style={styles.text}>No shared-genre path was found. The model can rank a movie without this evidence.</Text>}
    </ScrollView>
  </Modal></View>;
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0d1117' },
  content: { padding: 20, paddingTop: 60, paddingBottom: 50 },
  heading: { color: '#f8fafc', fontSize: 30, fontWeight: 'bold', marginBottom: 12 },
  subheading: { color: '#7dd3fc', fontSize: 22, fontWeight: 'bold', marginTop: 24, marginBottom: 12 },
  text: { color: '#aeb8c8', marginBottom: 12, lineHeight: 21 },
  input: { color: '#fff', borderColor: '#334155', borderWidth: 1, borderRadius: 8, padding: 12, marginBottom: 12 },
  button: { backgroundColor: '#7dd3fc', padding: 14, borderRadius: 8 },
  buttonText: { color: '#0d1117', fontWeight: 'bold', textAlign: 'center' },
  error: { color: '#fca5a5', marginTop: 16 },
  link: { color: '#7dd3fc', fontSize: 18, marginBottom: 24 },
  evidence: { color: '#f8fafc', backgroundColor: '#172033', padding: 16, borderRadius: 8, marginBottom: 12, lineHeight: 24 },
});
