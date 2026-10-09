"""CPU-only contracts; M4 numerical equivalence checked separately."""
import unittest

from orinth_clef.early_exit import check_depth
from orinth_clef.schema import SchemaError


class EarlyExitTests(unittest.TestCase):
    def test_depth_boundaries(self):
        check_depth(1, 36)
        check_depth(18, 36)
        check_depth(36, 36)
        for invalid in (0, -1, 37, 18.0, True, "18"):
            with self.assertRaises(SchemaError):
                check_depth(invalid, 36)


if __name__ == "__main__":
    unittest.main()
