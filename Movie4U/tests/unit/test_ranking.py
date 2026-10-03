import numpy as np
from movie4u.recommender.evaluate import top_indices, ranking_metrics
from movie4u.recommender.train import sample_negatives


def test_ranking_excludes_history_and_breaks_ties():
    assert top_indices([1, 9, 1, 0], {1}, 3).tolist() == [0, 2, 3]


def test_negative_samples_exclude_all_training_ratings():
    pairs = sample_negatives([0] * 20, 3, {0: {0, 1}}, np.random.default_rng(42))
    assert set(pairs[:, 1]) == {2}


def test_perfect_ranking():
    class Dataset:
        movies = list(range(3))
        users = [0]
        observed = {0: {2}}

        def relevant(self, split):
            return {0: {0}}

    result = ranking_metrics(np.array([[3, 1, 5]]), Dataset(), k=1)
    assert result["ndcg@1"] == 1
    assert result["recall@1"] == 1
