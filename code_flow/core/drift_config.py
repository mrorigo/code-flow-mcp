"""Default configuration values for drift detection."""

DEFAULT_DRIFT_ENABLED = False
DEFAULT_DRIFT_GRANULARITY = "module"  # module | file
DEFAULT_DRIFT_MIN_ENTITY_SIZE = 3
DEFAULT_DRIFT_CLUSTER_ALGORITHM = "hdbscan"
DEFAULT_DRIFT_CLUSTER_EPS = 0.75
DEFAULT_DRIFT_CLUSTER_MIN_SAMPLES = 5
DEFAULT_DRIFT_NUMERIC_FEATURES = [
    "complexity_mean",
    "complexity_variance",
    "nloc_mean",
    "nloc_variance",
    "decorator_count_mean",
    "dependency_count_mean",
    "exception_count_mean",
    "incoming_degree_mean",
    "outgoing_degree_mean",
]
DEFAULT_DRIFT_TEXTUAL_FEATURES = [
    "decorators",
    "external_dependencies",
    "catches_exceptions",
]
DEFAULT_DRIFT_IGNORE_PATH_PATTERNS = ["**/tests/**", "**/__pycache__/**", "**/node_modules/**"]
DEFAULT_DRIFT_CONFIDENCE_THRESHOLD = 0.6
