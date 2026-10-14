"""Group-disjoint key/value retrieval data with reproducible compositional splits.

Each *mapping* consists of four distinct keys drawn from eight, each assigned a
binary value. All permutations and query variants of a mapping belong to one
split. The split therefore excludes every test mapping from training, not just
identical sequence strings. This is controlled synthetic compositional testing,
not evidence about open-ended reasoning.
"""
from functools import lru_cache
from hashlib import blake2s
from itertools import combinations, product

import numpy as np

KEY_COUNT = 8
PAIR_COUNT = 4
VALUE_START = KEY_COUNT
MARKER = KEY_COUNT + 2
VOCAB_SIZE = KEY_COUNT + 3
SEQ_LEN = 2 * PAIR_COUNT + 2


def mapping_bucket(mapping):
    """Stable five-way bucket on canonical key/bit pairs; bucket zero is test."""
    raw = bytes(v for pair in mapping for v in pair)
    return int.from_bytes(blake2s(raw, digest_size=4).digest(), "big") % 5


@lru_cache(maxsize=1)
def mapping_pools():
    train, test = [], []
    for keys in combinations(range(KEY_COUNT), PAIR_COUNT):
        for bits in product((0, 1), repeat=PAIR_COUNT):
            mapping = tuple(zip(keys, bits))
            (test if mapping_bucket(mapping) == 0 else train).append(mapping)
    return (np.asarray(train, dtype=np.int32),
            np.asarray(test, dtype=np.int32))


def canonical_mapping(tokens):
    """Identity of a sequence's underlying map (ignores presentation and query)."""
    tokens = np.asarray(tokens)
    return tuple(sorted((int(tokens[i]), int(tokens[i + 1] - VALUE_START))
                        for i in range(0, 2 * PAIR_COUNT, 2)))


def sample(rng, mappings, batch):
    """Fresh presentation and query for a uniformly sampled canonical mapping."""
    if batch <= 0:
        raise ValueError("batch must be positive")
    selected = mappings[rng.integers(len(mappings), size=batch)]
    order = np.argsort(rng.random((batch, PAIR_COUNT)), axis=1)
    pairs = selected[np.arange(batch)[:, None], order]
    query_pos = rng.integers(PAIR_COUNT, size=batch)
    row = np.arange(batch)
    tokens = np.empty((batch, SEQ_LEN), dtype=np.int32)
    tokens[:, 0:2 * PAIR_COUNT:2] = pairs[:, :, 0]
    tokens[:, 1:2 * PAIR_COUNT:2] = pairs[:, :, 1] + VALUE_START
    tokens[:, -2] = MARKER
    tokens[:, -1] = pairs[row, query_pos, 0]
    labels = pairs[row, query_pos, 1].copy()
    return tokens, labels


def describe_split():
    train, test = mapping_pools()
    return {"group_definition": "unordered four-key -> binary-value mapping",
            "split_rule": "blake2s(canonical mapping bytes) modulo 5; bucket 0=test",
            "train_mappings": len(train), "test_mappings": len(test),
            "total_mappings": len(train) + len(test),
            "possible_presentations_per_mapping": 24 * PAIR_COUNT,
            "vocab_size": VOCAB_SIZE, "sequence_length": SEQ_LEN}
