"""Statistics of comparing two runs: is the difference real or is it noise?"""

from ranklens.stats.bootstrap import MIN_QUERIES, paired_bootstrap, paired_deltas
from ranklens.stats.compare import compare
from ranklens.stats.permutation import permutation_test

__all__ = ["MIN_QUERIES", "compare", "paired_bootstrap", "paired_deltas", "permutation_test"]
