"""Central configuration constants for the EML framework.

All magic numbers and experiment parameters live here. Import from this module
rather than hard-coding values in individual files. This ensures reproducibility
and makes tuning experiments a single-place change.
"""

# ---------------------------------------------------------------------------
# Numeric validation defaults
# ---------------------------------------------------------------------------

#: Default number of sample points used in numeric domain validation.
DEFAULT_SAMPLE_COUNT: int = 100

#: Default lower bound for numeric validation sample range.
DEFAULT_SAMPLE_LOW: float = 0.1

#: Default upper bound for numeric validation sample range.
DEFAULT_SAMPLE_HIGH: float = 5.0

# ---------------------------------------------------------------------------
# Weighted cost coefficients (symbolic circuit cost model)
# ---------------------------------------------------------------------------

#: Cost assigned to each EML primitive node in the weighted circuit cost.
EML_OPERATOR_COST: int = 3

#: Cost assigned to each raw exp/log node in the weighted circuit cost.
EXP_LOG_OPERATOR_COST: int = 2

#: Cost assigned to Abs nodes.
ABS_OPERATOR_COST: int = 1

#: Cost assigned to Add nodes.
ADD_OPERATOR_COST: int = 1

#: Cost assigned to Mul nodes.
MUL_OPERATOR_COST: int = 2

#: Cost assigned to Pow nodes.
POW_OPERATOR_COST: int = 3

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Global random seed for all experiments and searches.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Symbolic regression / evolutionary search defaults
# ---------------------------------------------------------------------------

#: Total number of candidate evaluations per search run.
SEARCH_BUDGET: int = 4000

#: Population size in the evolutionary symbolic regression search.
SEARCH_POP_SIZE: int = 100

#: Number of elite candidates carried over each generation.
SEARCH_ELITE_SIZE: int = 15

#: Maximum expression tree depth for candidate generation.
#: Set to 6 to reduce representational bias against the EML grammar,
#: which requires deeper structures to encode exp/log operations.
SEARCH_MAX_DEPTH: int = 6

#: Number of sample points used for MSE evaluation in symbolic regression.
SEARCH_SAMPLE_POINTS: int = 201

# ---------------------------------------------------------------------------
# Benchmark / validation
# ---------------------------------------------------------------------------

#: Default number of sample points for benchmark numeric validation.
BENCHMARK_SAMPLE_POINTS: int = 301
