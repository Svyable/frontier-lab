"""CPU contracts for experimental in-place KV rollback preconditions."""
import unittest

from orinth_clef.rollback_trie import restore_offsets, snapshot_offsets
from orinth_clef.schema import SchemaError


class Cache:
    def __init__(self, offset):
        self.offset = offset
        self.keys = None
        self.values = None


class RollbackContracts(unittest.TestCase):
    def test_snapshot_restore(self):
        cache = [Cache(11), Cache(11)]
        before = snapshot_offsets(cache)
        cache[0].offset = 22
        cache[1].offset = 23
        restore_offsets(cache, before)
        self.assertEqual(snapshot_offsets(cache), (11, 11))

    def test_invalid_cache_fails_closed(self):
        for cache in ([], None, [object()], [Cache(0), {"offset": 0}]):
            with self.subTest(cache=cache):
                with self.assertRaises(SchemaError):
                    snapshot_offsets(cache)

    def test_restore_layer_mismatch(self):
        with self.assertRaisesRegex(SchemaError, "layer count"):
            restore_offsets([Cache(1)], (1, 2))


if __name__ == "__main__":
    unittest.main()
