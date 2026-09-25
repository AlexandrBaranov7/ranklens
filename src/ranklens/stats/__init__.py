"""Statistics of comparing two runs: is the difference real or is it noise?

**Which hypothesis.** Everything here is about one number: the mean over queries of the
per-query difference of **one metric at one cutoff** (for example NDCG@10), B minus A.

- `permutation_test` tests H0: the per-query differences are symmetric around zero,
  that is, for every query it makes no difference which run is called A and which B.
  Rejecting H0 means such a mean difference is unlikely to come from chance alone.
- `paired_bootstrap` tests nothing: it says how precisely that mean is measured
  if the queries at hand are a sample of a larger population of queries.
- `benjamini_hochberg` keeps many comparisons honest: over 40 segments a couple of
  "significant" ones appear by chance, so reports show q-values.
- `minimum_detectable_effect` answers the question to ask first: is this many queries
  enough to see the effect we care about?
- The unit of observation is the **query**. The inference is about the population of
  queries, not about documents, sessions or users.

**What these do not answer.** Not "do the two rankings differ": two runs can put documents
in completely different orders and still have the same NDCG — for the orders themselves
use `ranklens.metrics.rbo`. And not "will users notice": that is an online experiment.
"""

from ranklens.stats.bootstrap import MIN_QUERIES, paired_bootstrap, paired_deltas
from ranklens.stats.compare import compare
from ranklens.stats.multiple import adjust, benjamini_hochberg
from ranklens.stats.permutation import permutation_test
from ranklens.stats.power import minimum_detectable_effect, required_queries, standard_deviation

__all__ = [
    "MIN_QUERIES",
    "adjust",
    "benjamini_hochberg",
    "compare",
    "minimum_detectable_effect",
    "paired_bootstrap",
    "paired_deltas",
    "permutation_test",
    "required_queries",
    "standard_deviation",
]
