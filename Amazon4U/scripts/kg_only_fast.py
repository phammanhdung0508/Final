"""Batched validation scoring for the frozen KG-only and Popularity baselines.

`recommendation_core.KGOnlyScorer` builds the IDF-weighted category and brand profiles
(docs/kg-only-scoring.md) and scores one user and one item at a time. This module reuses
those exact profiles and gives the same scores, bit for bit:

- products with the same category set have the same category vector, so the category
  cosine is computed once per user and distinct set, with `math.fsum` over the same
  products as the reference. Plain floating-point sums would break the reference's
  exact ties between different sets;
- each product has at most one brand, so the brand cosine is the user's weight at
  the product's brand;
- the score is `alpha * category + beta * brand`, elementwise in float64;
- a user with positive history but no usable positively weighted channel gets the
  whole-user `unusable_metadata_profile` popularity fallback.

Profiles do not depend on the mixture, so one `FastKGOnly` serves every mixture.
`EXTENDED_GRID` adds the two brand-heavier mixtures of the labeled post-validation
variant (docs/kg-only-extended-brand-amendment.md); the core grid is unchanged.

KG-v2 (docs/kg-v2-enrichment-plan.md) adds an optional attribute channel, built exactly
like the category channel: a `KGOnlyScorer` given the attributes as its categories
yields their IDF, product and user vectors. Its mixtures are the 3-tuples of
`KG_V2_GRID`, scored `alpha * category + beta * brand + gamma * attribute`.
Attempt C adds a two-sided `accessory` channel (scripts/kg_accessory.py): user vectors
come from the history-side tokens, product vectors from the candidate-side tokens, each
built by its own `KGOnlyScorer`. Its 4-tuple mixtures are `KG_C_GRID`, with the extra
term `+ delta * accessory`.
Ranking, filtering and metrics are `vectorized_evaluation.evaluate_scores`. Only the
validation split is supported. Final-test evaluation is not authorized.
"""

import math

import torch

from scripts.recommendation_core import KGOnlyScorer
from scripts.vectorized_evaluation import evaluate_scores

DTYPE = torch.float64
EXTENDED_GRID = KGOnlyScorer.GRID + ((0.25, 0.75), (0.0, 1.0))
KG_V2_GRID = ((0.5, 0.5, 0.0), (0.4, 0.4, 0.2), (1 / 3, 1 / 3, 1 / 3))
KG_C_GRID = ((0.4, 0.4, 0.2, 0.0), (0.35, 0.35, 0.15, 0.15), (0.3, 0.3, 0.15, 0.25))


def reference_scorer(context, item_categories, item_brands, mixture):
    """The accepted per-user scorer at any `EXTENDED_GRID` mixture.

    `KGOnlyScorer` accepts only the core grid. Its profiles do not depend on the
    mixture and its score is `alpha * category + beta * brand`, so the variant
    mixtures set those two weights on a core-grid instance.
    """
    if mixture not in EXTENDED_GRID:
        raise ValueError(f'Invalid mixture {mixture}. Must be one of {EXTENDED_GRID}.')
    scorer = KGOnlyScorer(context, item_categories, item_brands)
    scorer.mixture = mixture
    scorer.alpha, scorer.beta = mixture
    return scorer


class ReferenceKGv2:
    """Per-user KG-only scorer with the attribute (and accessory) channels, written like `KGOnlyScorer`.

    The spot-check oracle for `KG_V2_GRID` and `KG_C_GRID`. Category and brand vectors
    come from `KGOnlyScorer`; the attribute vectors come from a second `KGOnlyScorer`
    given the attributes as categories; the accessory user and product vectors from two
    more, given the history-side and candidate-side tokens. Each cosine is `math.fsum`
    over sorted shared keys, zero when a channel is unusable, as in the reference.
    """

    def __init__(self, context, item_categories, item_brands, item_attributes, mixture, accessory=None):
        if mixture not in KG_V2_GRID + (KG_C_GRID if accessory else ()):
            raise ValueError(f'Invalid mixture {mixture} for these channels.')
        self.context = context
        self.weights = mixture
        base = KGOnlyScorer(context, item_categories, item_brands)
        attr = KGOnlyScorer(context, item_attributes, {})
        self.users = [base.user_cat_vec, base.user_brand_vec, attr.user_cat_vec]
        self.products = [base.product_cat_vec, base.product_brand_vec, attr.product_cat_vec]
        if accessory:
            history, candidate = accessory
            self.users.append(KGOnlyScorer(context, history, {}).user_cat_vec)
            self.products.append(KGOnlyScorer(context, candidate, {}).product_cat_vec)

    def score_candidates(self, user, candidate_items):
        if not self.context.positive_history.get(user):
            return ({i: float(self.context.popularity.get(i, 0)) for i in candidate_items},
                    'fallback', 'empty_positive_history')
        profiles = [vec.get(user, {}) for vec in self.users]
        usable = [w > 0.0 and len(p) > 0 for w, p in zip(self.weights, profiles)]
        if not any(usable):
            return ({i: float(self.context.popularity.get(i, 0)) for i in candidate_items},
                    'fallback', 'unusable_metadata_profile')
        scores = {}
        for item in candidate_items:
            cos = []
            for ok, u, products in zip(usable, profiles, self.products):
                p = products.get(item, {})
                cos.append(math.fsum(u[k] * p[k] for k in sorted(set(u) & set(p))) if ok and p else 0.0)
            score = self.weights[0] * cos[0] + self.weights[1] * cos[1] + self.weights[2] * cos[2]
            if len(self.weights) == 4:
                score = score + self.weights[3] * cos[3]
            scores[item] = score
        return scores, 'personalized', None


class _SetChannel:
    """Cosine between user vectors and multi-attribute product vectors.

    Products with the same attribute set have the same vector, so the cosine is
    computed once per user and distinct set, with `math.fsum` as in the reference.
    """

    def __init__(self, items, product_vec, user_vec):
        self.user_vec = user_vec
        sets, item_set = {}, []
        for item in items:
            key = tuple(sorted(product_vec[item].items()))
            item_set.append(sets.setdefault(key, len(sets)))
        self.item_set = torch.tensor(item_set)
        self.containing = {}
        for key, s in sets.items():
            for attr, v in key:
                self.containing.setdefault(attr, []).append((s, v))
        self.n_sets = len(sets)
        self._rows = {}

    def row(self, user):
        row = self._rows.get(user)
        if row is None:
            terms = [[] for _ in range(self.n_sets)]
            for attr, u in self.user_vec.get(user, {}).items():
                for s, p in self.containing.get(attr, ()):
                    terms[s].append(u * p)
            row = self._rows[user] = torch.tensor([math.fsum(t) for t in terms], dtype=DTYPE)
        return row

    def scores(self, users):
        return torch.stack([self.row(u) for u in users])[:, self.item_set]


class FastKGOnly:
    def __init__(self, context, item_categories, item_brands, item_attributes=None, accessory=None):
        scorer = KGOnlyScorer(context, item_categories, item_brands)
        self.context = context
        self.items = context.sorted_warm_items
        self.category = _SetChannel(self.items, scorer.product_cat_vec, scorer.user_cat_vec)
        self.attribute = None
        if item_attributes is not None:
            attr = KGOnlyScorer(context, item_attributes, {})
            self.attribute = _SetChannel(self.items, attr.product_cat_vec, attr.user_cat_vec)
        self.accessory = None
        if accessory is not None:
            history, candidate = accessory
            self.accessory = _SetChannel(self.items, KGOnlyScorer(context, candidate, {}).product_cat_vec,
                                         KGOnlyScorer(context, history, {}).user_cat_vec)
        self.user_brand = scorer.user_brand_vec
        brands = {b: i for i, b in enumerate(sorted(scorer.idf_brand))}
        self.brands = brands
        # Index 0 is "no brand", whose user weight is always zero.
        self.item_brand = torch.tensor([1 + brands[next(iter(scorer.product_brand_vec[i]))]
                                        if scorer.product_brand_vec[i] else 0 for i in self.items])

    def ranker(self, mixture):
        """`(score_users, route_of)` for `evaluate_scores` under one frozen mixture."""
        allowed = EXTENDED_GRID + (KG_V2_GRID if self.attribute else ()) \
            + (KG_C_GRID if self.attribute and self.accessory else ())
        if mixture not in allowed:
            raise ValueError(f'Invalid mixture {mixture} for this graph.')
        alpha, beta, gamma, delta = (*mixture, 0.0, 0.0)[:4]
        channels = ((alpha, self.category.user_vec), (beta, self.user_brand),
                    (gamma, self.attribute.user_vec if self.attribute else {}),
                    (delta, self.accessory.user_vec if self.accessory else {}))

        def route_of(user):
            usable = any(w > 0.0 and vec.get(user) for w, vec in channels)
            return ('personalized', None) if usable else ('fallback', 'unusable_metadata_profile')

        def score_users(users):
            brand = torch.zeros(len(users), 1 + len(self.brands), dtype=DTYPE)
            for j, u in enumerate(users):
                for b, v in self.user_brand.get(u, {}).items():
                    brand[j, 1 + self.brands[b]] = v
            score = alpha * self.category.scores(users) + beta * brand[:, self.item_brand]
            if len(mixture) >= 3:
                score = score + gamma * self.attribute.scores(users)
            if len(mixture) == 4:
                score = score + delta * self.accessory.scores(users)
            return score
        return score_users, route_of


def evaluate_kg_only(fast, mixture, targets, *, chunk_users=256):
    score_users, route_of = fast.ranker(mixture)
    return evaluate_scores(fast.context, fast.items, score_users, targets, route_of=route_of,
                           chunk_users=chunk_users)


def evaluate_popularity(context, targets, *, chunk_users=256):
    items = context.sorted_warm_items
    popularity = torch.tensor([float(context.popularity.get(p, 0)) for p in items], dtype=DTYPE)
    return evaluate_scores(context, items, lambda users: popularity.expand(len(users), -1), targets,
                           route_of=lambda u: ('non_personalized', None), chunk_users=chunk_users)
