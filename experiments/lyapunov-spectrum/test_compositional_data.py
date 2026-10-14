"""Pure-numpy reproducibility and group-leakage tests (no Metal required)."""
import unittest

import numpy as np

from compositional_data import (KEY_COUNT, MARKER, PAIR_COUNT, VALUE_START,
                                canonical_mapping, describe_split, mapping_bucket,
                                mapping_pools, sample)


class CompositionalDataTests(unittest.TestCase):
    def test_partition_disjoint_and_complete(self):
        train, test = mapping_pools()
        tr = set(map(lambda x: tuple(map(tuple, x)), train))
        te = set(map(lambda x: tuple(map(tuple, x)), test))
        self.assertFalse(tr.intersection(te))
        self.assertEqual(len(tr) + len(te), 1120)
        self.assertGreater(len(te), 150)
        self.assertTrue(all(mapping_bucket(x) == 0 for x in te))
        self.assertTrue(all(mapping_bucket(x) != 0 for x in tr))

    def test_sequence_labels_and_no_leakage(self):
        train, test = mapping_pools()
        for pool, expected_bucket in ((train, False), (test, True)):
            x, y = sample(np.random.default_rng(45), pool, 2500)
            self.assertEqual(x.shape, (2500, 10))
            self.assertEqual(y.shape, (2500,))
            self.assertTrue(np.all(x[:, -2] == MARKER))
            self.assertTrue(set(np.unique(y)).issubset({0, 1}))
            for row, label in zip(x, y):
                keys = row[0:-2:2]
                bits = row[1:-2:2] - VALUE_START
                self.assertEqual(len(set(keys.tolist())), PAIR_COUNT)
                self.assertTrue(np.all(keys < KEY_COUNT))
                self.assertTrue(np.all((bits == 0) | (bits == 1)))
                self.assertEqual(int(label), int(bits[np.where(keys == row[-1])[0][0]]))
                self.assertEqual(mapping_bucket(canonical_mapping(row)) == 0,
                                 expected_bucket)

    def test_determinism_and_stats(self):
        train, _ = mapping_pools()
        x1, y1 = sample(np.random.default_rng(1212), train, 64)
        x2, y2 = sample(np.random.default_rng(1212), train, 64)
        self.assertTrue(np.array_equal(x1, x2))
        self.assertTrue(np.array_equal(y1, y2))
        stats = describe_split()
        self.assertEqual(stats["total_mappings"], 1120)
        self.assertEqual(stats["possible_presentations_per_mapping"], 96)
        self.assertEqual(stats["vocab_size"], 11)
        with self.assertRaises(ValueError):
            sample(np.random.default_rng(2), train, 0)


if __name__ == "__main__":
    unittest.main()
