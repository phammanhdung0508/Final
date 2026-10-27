"""Attempt C: accessory roles with the audit fixes, tokens, and the KG-only accessory channel."""

import random
import unittest

from scripts.kg_accessory import accessory_tokens, roles
from scripts.kg_only_fast import KG_C_GRID, FastKGOnly, ReferenceKGv2, evaluate_kg_only
from scripts.recommendation_evaluation import TrainingContext
from test_kg_only_fast import attributes_of, world

GUITAR = ['Musical Instruments', 'Guitars', 'Acoustic Guitars']


class RoleTests(unittest.TestCase):
    def test_an_instrument_serves_nothing(self):
        self.assertEqual(roles('Vangoa Acoustic Guitar for Beginner Adult, Acoustic Guitar Starter', GUITAR, None),
                         ({'acoustic guitar', 'guitar'}, set()))

    def test_a_comma_list_after_for_counts_every_family(self):
        _, serves = roles('Clip On Tuner For All Instruments, Ukulele, Guitar, Bass, Violin; Black', [], None)
        self.assertEqual(serves, {'ukulele', 'guitar', 'bass guitar', 'orchestral strings'})

    def test_a_phrase_stops_at_a_semicolon_or_bracket(self):
        self.assertEqual(roles('Strap for Banjo; fits Guitar', [], None)[1], {'banjo'})
        self.assertEqual(roles('Strap for Banjo (Guitar not included)', [], None)[1], {'banjo'})

    def test_tokens_flip_between_history_and_candidate(self):
        history, candidate = accessory_tokens({'g': ({'guitar'}, set()), 'capo': (set(), {'guitar'}), 'pick': (set(), set())})
        self.assertEqual(history, {'g': ['own:guitar'], 'capo': ['serve:guitar']})
        self.assertEqual(candidate, {'g': ['serve:guitar'], 'capo': ['own:guitar']})


def accessory_of(seed, items):
    rng = random.Random(seed)
    product_roles = {}
    for p in items:
        kind = rng.random()
        if kind < 0.25:
            product_roles[p] = ({rng.choice(['guitar', 'ukulele'])}, set())
        elif kind < 0.6:
            product_roles[p] = (set(), set(rng.sample(['guitar', 'ukulele', 'drums'], rng.randint(1, 2))))
    return accessory_tokens(product_roles)


class AccessoryChannelTests(unittest.TestCase):
    def test_a_hand_calculated_accessory_score(self):
        # u owns guitar g; the candidate capo serves guitar. History tokens: g {own:guitar};
        # candidate tokens: capo {own:guitar}. One token per side, so both vectors are 1.0 and
        # the accessory cosine is 1; the score is delta = 0.25 with the other channels empty.
        train = [('u', 'g', 5.0), ('v', 'capo', 5.0), ('v', 'pick', 5.0)]
        accessory = accessory_tokens({'g': ({'guitar'}, set()), 'capo': (set(), {'guitar'}), 'pick': (set(), set())})
        fast = FastKGOnly(TrainingContext(train), {}, {}, {}, accessory)
        row = dict(zip(fast.items, fast.ranker((0.3, 0.3, 0.15, 0.25))[0](['u'])[0].tolist()))
        self.assertEqual(row, {'capo': 0.25, 'g': 0.0, 'pick': 0.0})

    def test_scores_and_routes_equal_the_per_user_reference(self):
        for seed in (1, 2, 3):
            train, _, cats, brands = world(seed)
            items = {p for _, p, _ in train}
            attrs, accessory = attributes_of(seed, items), accessory_of(seed, items)
            for condition in ('all_ratings', 'rating_ge_4'):
                context = TrainingContext(train, condition=condition)
                fast = FastKGOnly(context, cats, brands, attrs, accessory)
                users = sorted(context.users)
                for mixture in KG_C_GRID:
                    reference = ReferenceKGv2(context, cats, brands, attrs, mixture, accessory)
                    score_users, route_of = fast.ranker(mixture)
                    for u, row in zip(users, score_users(users).tolist()):
                        expected, route, reason = reference.score_candidates(u, list(fast.items))
                        if route == 'personalized' or reason == 'unusable_metadata_profile':
                            self.assertEqual(route_of(u), (route, reason), (seed, condition, mixture, u))
                        if route == 'personalized':
                            self.assertEqual(row, [expected[p] for p in fast.items], (seed, condition, mixture, u))

    def test_zero_accessory_weight_reproduces_attempt_b(self):
        train, targets, cats, brands = world(2)
        items = {p for _, p, _ in train}
        context = TrainingContext(train)
        attrs = attributes_of(2, items)
        self.assertEqual(evaluate_kg_only(FastKGOnly(context, cats, brands, attrs, accessory_of(2, items)),
                                          (0.4, 0.4, 0.2, 0.0), targets),
                         evaluate_kg_only(FastKGOnly(context, cats, brands, attrs), (0.4, 0.4, 0.2), targets))

    def test_four_channel_mixtures_need_accessory_links(self):
        train, _, cats, brands = world(1)
        fast = FastKGOnly(TrainingContext(train), cats, brands, attributes_of(1, {p for _, p, _ in train}))
        with self.assertRaises(ValueError):
            fast.ranker((0.35, 0.35, 0.15, 0.15))


if __name__ == '__main__':
    unittest.main()
