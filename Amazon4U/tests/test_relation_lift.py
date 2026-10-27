"""Relation lift: linked and random rates, direction, and the instrument roles."""

import unittest

from scripts.relation_lift import instrument_roles, lift


class LiftTests(unittest.TestCase):
    def test_a_relation_that_always_holds_next_has_lift_one_over_its_random_rate(self):
        values = {'a': {'x'}, 'b': {'x'}, 'c': {'y'}, 'd': {'y'}}
        weights = {'a': 1, 'b': 1, 'c': 1, 'd': 1}
        stats = lift([(['a'], 'b')] * 200, values, weights, draws=50)
        self.assertEqual(stats['linked_rate'], 1.0)
        self.assertAlmostEqual(stats['random_rate'], 0.5, delta=0.05)
        self.assertAlmostEqual(stats['lift'], 2.0, delta=0.2)

    def test_events_without_a_history_value_are_skipped(self):
        stats = lift([(['z'], 'a'), (['a'], 'c')], {'a': {'x'}, 'c': {'y'}}, {'a': 1, 'c': 1})
        self.assertEqual(stats['events_with_history_value'], 1)
        self.assertEqual(stats['linked_rate'], 0.0)

    def test_direction_links_an_accessory_to_an_owned_instrument_only(self):
        history = {'guitar': {('own', 'g')}, 'capo': {('serve', 'g')}}
        nxt = {'guitar': {('serve', 'g')}, 'capo': {('own', 'g')}}
        weights = {'guitar': 1, 'capo': 1}
        self.assertEqual(lift([(['guitar'], 'capo')], history, weights, nxt)['linked_rate'], 1.0)
        self.assertEqual(lift([(['capo'], 'guitar')], history, weights, nxt)['linked_rate'], 1.0)
        self.assertEqual(lift([(['guitar'], 'guitar')], history, weights, nxt)['linked_rate'], 0.0)


class InstrumentRoleTests(unittest.TestCase):
    def test_roles_come_from_categories_titles_and_compatibility(self):
        self.assertEqual(instrument_roles('Fender Player Stratocaster', ['Musical Instruments', 'Guitars',
                                                                         'Electric Guitars'], None),
                         ({'electric guitar', 'guitar'}, set()))
        self.assertEqual(instrument_roles('Kyser Capo for Acoustic Guitars, Black', ['Instrument Accessories'], None)[1],
                         {'acoustic guitar', 'guitar'})
        self.assertEqual(instrument_roles('Pop Filter', [], 'Microphone, Smartphone')[1], {'microphone'})

    def test_words_outside_a_for_phrase_do_not_count(self):
        self.assertEqual(instrument_roles('Guitar Strap, Black Leather', [], None), (set(), set()))
        self.assertEqual(instrument_roles('Bassoon Reed', [], None), (set(), set()))
