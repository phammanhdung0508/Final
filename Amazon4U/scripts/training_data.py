"""Bounded, owned, in-memory H-REF input adapter. No filesystem acquisition."""
import copy
import math

from scripts.checkpoint_identity import (
    CanonicalResumeIdentity, validate_canonical_identity,
)
from scripts.checkpoint_payload import (
    preflight_resource_accounting, validate_resource_policy,
)
from scripts.recommendation_evaluation import TrainingContext


class HarnessError(ValueError):
    """Explicit H boundary, authorization, capability or execution failure."""


def checked_policy(policy):
    parsed = validate_resource_policy(policy)
    if (parsed['kind'] != 'synthetic'
            or parsed['max_input_bytes'] > 2097152
            or parsed['max_output_bytes'] > 2097152
            or parsed['max_container_depth'] > 64
            or parsed['max_visited_values'] > 16384):
        raise HarnessError('H-REF requires a bounded synthetic resource policy')
    return parsed


def preflight(value, policy, role):
    preflight_resource_accounting(value, checked_policy(policy), role=role)


def clean_id(value):
    if (type(value) is not str or not value or value.strip() != value
            or len(value) > 256):
        raise HarnessError('IDs must be bounded clean strings')
    try:
        value.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise HarnessError('invalid Unicode ID') from exc
    return value


def bounded_training_context(records, condition, resource_policy):
    """Validate before materializing the accepted training-only context."""
    preflight(records, resource_policy, 'H training records')
    if type(records) not in (list, tuple) or not 1 <= len(records) <= 96:
        raise HarnessError('training records must contain 1..96 owned records')
    users, products, seen, owned = set(), set(), set(), []
    for record in records:
        if (type(record) is not dict
                or not {'user_id', 'parent_asin', 'rating'} <= set(record)
                or set(record) - {'user_id', 'parent_asin', 'rating', 'timestamp'}):
            raise HarnessError('training record fields are not permitted')
        user = clean_id(record['user_id'])
        item = clean_id(record['parent_asin'])
        rating = record['rating']
        if type(rating) not in (int, float) or not math.isfinite(rating):
            raise HarnessError('rating must be a finite nonboolean number')
        if 'timestamp' in record and type(record['timestamp']) is not int:
            raise HarnessError('fixture timestamps must be exact integers')
        if (user, item) in seen:
            raise HarnessError('duplicate training pair')
        seen.add((user, item))
        users.add(user)
        products.add(item)
        owned.append(dict(record))
    if len(users) > 8 or len(products) > 12:
        raise HarnessError('fixture population exceeds 8 users / 12 products')
    return TrainingContext(owned, condition=condition)


class TrainingData:
    """A complete D1 input packet checked against a separate trusted expectation.

    Validation records are retained for evaluation, never passed to TrainingContext
    or graph construction. A caller must explicitly authorize synthetic execution;
    this adapter does not infer that authorization from the packet.
    """

    def __init__(self, identity_packet, *, expected_identity, resource_policy):
        policy = checked_policy(resource_policy)
        preflight(identity_packet, policy, 'H actual identity')
        preflight(expected_identity, policy, 'H expected identity')
        if type(identity_packet) is not dict or type(expected_identity) is not dict:
            raise HarnessError('identities must be serialized plain dictionaries')
        actual = CanonicalResumeIdentity.from_dict(identity_packet)
        expected = CanonicalResumeIdentity.from_dict(expected_identity)
        validate_canonical_identity(actual, expected)
        self.identity = CanonicalResumeIdentity.from_dict(actual.to_dict())
        ident = self.identity
        self.context = bounded_training_context(
            ident.training_records, ident.feedback_condition, policy)
        limits = {'User': 8, 'Product': 12, 'CategoryPath': 8, 'Brand': 8}
        if set(ident.graph_nodes) != set(limits):
            raise HarnessError('fixture requires exactly the four accepted node types')
        for kind, maximum in limits.items():
            if len(ident.graph_nodes[kind]) > maximum:
                raise HarnessError('typed fixture population exceeds its bound')
            for node in ident.graph_nodes[kind]:
                clean_id(node)
        if len(ident.graph_edges) > 512:
            raise HarnessError('fixture graph exceeds 512 directed edges')
        if len(ident.product_metadata) > 12:
            raise HarnessError('fixture metadata exceeds 12 products')
        self.product_categories = {}
        self.product_brands = {}
        for item, metadata in ident.product_metadata.items():
            clean_id(item)
            if set(metadata) != {'categories', 'brand'}:
                raise HarnessError('only CategoryPath and Brand metadata are permitted')
            categories = metadata['categories']
            if len(categories) > 8:
                raise HarnessError('too many category IDs per product')
            self.product_categories[item] = [clean_id(c) for c in categories]
            if metadata['brand'] is not None:
                self.product_brands[item] = clean_id(metadata['brand'])
        validation = ident.validation_inputs
        if type(validation) is not dict or set(validation) != {'targets', 'records'}:
            raise HarnessError('fixture validation requires actual records and target IDs')
        records = validation['records']
        if type(records) not in (list, tuple) or not 1 <= len(records) <= 8:
            raise HarnessError('fixture validation requires 1..8 records')
        users = set(self.context.users)
        items = set(self.context.all_training_items)
        training_pairs = {(r['user_id'], r['parent_asin']) for r in ident.training_records}
        seen_users = set()
        self.validation_records = []
        for record in records:
            if (type(record) is not dict
                    or set(record) != {'user_id', 'parent_asin', 'rating', 'timestamp'}):
                raise HarnessError('validation records require user,item,rating,timestamp')
            user = clean_id(record['user_id'])
            item = clean_id(record['parent_asin'])
            rating = record['rating']
            timestamp = record['timestamp']
            if (type(rating) not in (int, float) or not math.isfinite(rating)
                    or type(timestamp) is not int):
                raise HarnessError('invalid validation numeric fields')
            if user in seen_users or (user, item) in training_pairs:
                raise HarnessError('duplicate validation user or training/validation overlap')
            if list(validation['targets'].get(user, ())) != [item]:
                raise HarnessError('validation targets disagree with actual records')
            seen_users.add(user)
            users.add(user)
            items.add(item)
            self.validation_records.append((user, item, rating, timestamp))
        if set(validation['targets']) != seen_users:
            raise HarnessError('validation target population is incomplete')
        if len(users) > 8 or len(items) > 12:
            raise HarnessError('combined fixture population exceeds its bound')
        self.positive_pairs = tuple(sorted(
            (user, item)
            for user, positives in self.context.positive_history.items()
            for item in positives))
        self.resource_policy = copy.deepcopy(policy)

    def owned_copy(self, *, expected_identity):
        return TrainingData(self.identity.to_dict(), expected_identity=expected_identity,
                            resource_policy=self.resource_policy)
