"""Pure CPU trie structural contracts; MLX parity is tested on the M4."""
import unittest

from orinth_clef.prefix_trie import build_trie, trie_stats
from orinth_clef.schema import SchemaError


class PrefixTrieTests(unittest.TestCase):
    def test_shared_prefix_saves_nodes(self):
        candidates = {"first": [1, 2, 3], "second": [1, 2, 4], "third": [1, 5]}
        stats = trie_stats(build_trie(candidates))
        self.assertEqual(stats, {"unique_prefix_tokens": 5, "candidates": 3})
        self.assertLess(stats["unique_prefix_tokens"], sum(map(len, candidates.values())))

    def test_terminal_prefix_and_longer_candidate(self):
        root = build_trie({"short": [2], "long": [2, 3]})
        self.assertEqual(root["children"][2]["labels"], ["short"])
        self.assertEqual(root["children"][2]["children"][3]["labels"], ["long"])

    def test_duplicate_token_sequences_preserve_labels(self):
        root = build_trie({"alias_a": [5], "alias_b": [5]})
        self.assertEqual(root["children"][5]["labels"], ["alias_a", "alias_b"])

    def test_reject_empty_and_invalid(self):
        for candidates in ({}, {"empty": []}, {"negative": [-1]}, {"float": [1.5]}):
            with self.subTest(candidates=candidates):
                with self.assertRaises(SchemaError):
                    build_trie(candidates)


if __name__ == "__main__":
    unittest.main()
