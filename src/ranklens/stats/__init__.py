"""Statistics of comparing two runs: is the difference real or is it noise?"""

from ranklens.stats.bootstrap import MIN_QUERIES, paired_bootstrap, paired_deltas

__all__ = ["MIN_QUERIES", "paired_bootstrap", "paired_deltas"]
