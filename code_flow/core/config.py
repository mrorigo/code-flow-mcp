import os
import yaml
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

from code_flow.core.drift_config import (
    DEFAULT_DRIFT_ENABLED,
    DEFAULT_DRIFT_GRANULARITY,
    DEFAULT_DRIFT_MIN_ENTITY_SIZE,
    DEFAULT_DRIFT_CLUSTER_ALGORITHM,
    DEFAULT_DRIFT_CLUSTER_EPS,
    DEFAULT_DRIFT_CLUSTER_MIN_SAMPLES,
    DEFAULT_DRIFT_NUMERIC_FEATURES,
    DEFAULT_DRIFT_TEXTUAL_FEATURES,
    DEFAULT_DRIFT_IGNORE_PATH_PATTERNS,
    DEFAULT_DRIFT_CONFIDENCE_THRESHOLD,
)

# Default configuration values
DEFAULT_CONFIG_PATH = "codeflow.config.yaml"
DEFAULT_WATCH_DIRECTORIES = ["."]
DEFAULT_IGNORED_PATTERNS = ["venv", "**/__pycache__", ".git", ".idea", ".vscode", "node_modules"]
DEFAULT_CHROMADB_PATH = "./code_vectors_chroma"
DEFAULT_PROJECT_ROOT = None
DEFAULT_CODEFLOW_DIR = ".codeflow"
DEFAULT_MAX_GRAPH_DEPTH = 3
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MAX_TOKENS = 256
DEFAULT_LANGUAGE = "python"
DEFAULT_MIN_SIMILARITY = 0.1
DEFAULT_CALL_GRAPH_CONFIDENCE_THRESHOLD = 0.8

# Cortex memory defaults
DEFAULT_MEMORY_ENABLED = True
DEFAULT_MEMORY_COLLECTION = "cortex_memory_v1"
DEFAULT_MEMORY_SIMILARITY_WEIGHT = 0.7
DEFAULT_MEMORY_SCORE_WEIGHT = 0.3
DEFAULT_MEMORY_MIN_SCORE = 0.1
DEFAULT_MEMORY_CLEANUP_INTERVAL_SECONDS = 3600
DEFAULT_MEMORY_GRACE_SECONDS = 86400
DEFAULT_MEMORY_HALF_LIFE_DAYS = {
    "TRIBAL": 180.0,
    "EPISODIC": 7.0,
    "FACT": 30.0,
}
DEFAULT_MEMORY_DECAY_FLOOR = {
    "TRIBAL": 0.1,
    "EPISODIC": 0.01,
    "FACT": 0.05,
}

# Cortex memory resources (MCP)
DEFAULT_MEMORY_RESOURCES_ENABLED = True
DEFAULT_MEMORY_RESOURCES_LIMIT = 10
DEFAULT_MEMORY_RESOURCES_FILTERS: Dict[str, Any] = {}


class CodeFlowConfig(BaseModel):
    """
    Central configuration model for CodeFlowGraph.
    """
    watch_directories: List[str] = Field(default_factory=lambda: DEFAULT_WATCH_DIRECTORIES)
    ignored_patterns: List[str] = Field(default_factory=lambda: DEFAULT_IGNORED_PATTERNS)
    project_root: Optional[str] = Field(default=DEFAULT_PROJECT_ROOT)
    chromadb_path: str = Field(default=DEFAULT_CHROMADB_PATH)
    max_graph_depth: int = Field(default=DEFAULT_MAX_GRAPH_DEPTH)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS)
    language: str = Field(default=DEFAULT_LANGUAGE)
    min_similarity: float = Field(default=DEFAULT_MIN_SIMILARITY)
    call_graph_confidence_threshold: float = Field(default=DEFAULT_CALL_GRAPH_CONFIDENCE_THRESHOLD)
    
    # Optional LLM config for summaries (can be expanded)
    summary_generation_enabled: bool = False
    llm_config: Dict[str, Any] = Field(default_factory=dict)

    # Cortex memory
    memory_enabled: bool = DEFAULT_MEMORY_ENABLED
    memory_collection_name: str = DEFAULT_MEMORY_COLLECTION
    memory_similarity_weight: float = DEFAULT_MEMORY_SIMILARITY_WEIGHT
    memory_score_weight: float = DEFAULT_MEMORY_SCORE_WEIGHT
    memory_min_score: float = DEFAULT_MEMORY_MIN_SCORE
    memory_cleanup_interval_seconds: int = DEFAULT_MEMORY_CLEANUP_INTERVAL_SECONDS
    memory_grace_seconds: int = DEFAULT_MEMORY_GRACE_SECONDS
    memory_half_life_days: Dict[str, float] = Field(default_factory=lambda: DEFAULT_MEMORY_HALF_LIFE_DAYS.copy())
    memory_decay_floor: Dict[str, float] = Field(default_factory=lambda: DEFAULT_MEMORY_DECAY_FLOOR.copy())

    # Cortex memory resources (MCP)
    memory_resources_enabled: bool = DEFAULT_MEMORY_RESOURCES_ENABLED
    memory_resources_limit: int = DEFAULT_MEMORY_RESOURCES_LIMIT
    memory_resources_filters: Dict[str, Any] = Field(default_factory=lambda: DEFAULT_MEMORY_RESOURCES_FILTERS.copy())

    # Drift detection
    drift_enabled: bool = DEFAULT_DRIFT_ENABLED
    drift_granularity: str = DEFAULT_DRIFT_GRANULARITY
    drift_min_entity_size: int = DEFAULT_DRIFT_MIN_ENTITY_SIZE
    drift_cluster_algorithm: str = DEFAULT_DRIFT_CLUSTER_ALGORITHM
    drift_cluster_eps: float = DEFAULT_DRIFT_CLUSTER_EPS
    drift_cluster_min_samples: int = DEFAULT_DRIFT_CLUSTER_MIN_SAMPLES
    drift_numeric_features: List[str] = Field(default_factory=lambda: list(DEFAULT_DRIFT_NUMERIC_FEATURES))
    drift_textual_features: List[str] = Field(default_factory=lambda: list(DEFAULT_DRIFT_TEXTUAL_FEATURES))
    drift_ignore_path_patterns: List[str] = Field(default_factory=lambda: list(DEFAULT_DRIFT_IGNORE_PATH_PATTERNS))
    drift_confidence_threshold: float = DEFAULT_DRIFT_CONFIDENCE_THRESHOLD
    
    # Allow extra fields for flexibility
    class Config:
        extra = "allow"

    def require_project_root(self) -> Path:
        if not self.project_root:
            raise ValueError("project_root must be set in config to resolve .codeflow paths")
        return Path(self.project_root).resolve()

    def codeflow_dir(self) -> Path:
        return self.require_project_root() / DEFAULT_CODEFLOW_DIR

    def chroma_dir(self) -> Path:
        return self.codeflow_dir() / "chroma"

    def memory_dir(self) -> Path:
        return self.codeflow_dir() / "memory"

    def reports_dir(self) -> Path:
        return self.codeflow_dir() / "reports"

    def cache_dir(self) -> Path:
        return self.codeflow_dir() / "cache"


def load_config(
    config_path: Optional[str] = None, 
    cli_args: Optional[Dict[str, Any]] = None
) -> CodeFlowConfig:
    """
    Load configuration from file and overrides.
    
    Priority:
    1. CLI Arguments (if provided and not None)
    2. Config File (if provided or found at default path)
    3. Default Values
    
    Args:
        config_path: Path to the YAML config file. If None, tries 'codeflow.config.yaml'.
        cli_args: Dictionary of CLI arguments to override config values.
        
    Returns:
        CodeFlowConfig: The resolved configuration object.
    """
    config_data = {}
    
    # 1. Load from file
    # Determine path: explicit arg > default file in cwd > None
    target_path = config_path if config_path else DEFAULT_CONFIG_PATH
    path_obj = Path(target_path)
    
    if path_obj.exists() and path_obj.is_file():
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                file_data = yaml.safe_load(f)
                if file_data:
                    config_data.update(file_data)
            logging.info(f"Loaded configuration from {target_path}")
        except Exception as e:
            logging.warning(f"Failed to load config file {target_path}: {e}")
    elif config_path:
        # If user explicitly provided a path that doesn't exist, warn them
        logging.warning(f"Config file not found at explicit path: {config_path}")
    else:
        logging.info(f"No config file found at {DEFAULT_CONFIG_PATH}, using defaults.")

    # 2. Override with CLI args
    if cli_args:
        # Only override if the value is not None (and maybe check for default values if we had a way to know)
        # For now, we assume the caller filters out 'None' or we check here.
        # Common CLI args mapping to config fields:
        
        # Map 'directory' (CLI) to 'watch_directories' (Config)
        if cli_args.get('directory'):
            # If directory is '.', we might want to respect the config file's watch_directories if it exists
            # But usually CLI arg wins. Let's say if user typed a dir, it wins.
            # If it's the default '.', we might be careful. 
            # However, standard CLI behavior is explicit arg wins.
            # We'll handle this mapping in the caller or here if we want strict unification.
            # Let's keep it simple: direct mapping for now, caller handles complex logic if needed.
            pass

        for key, value in cli_args.items():
            if value is not None:
                config_data[key] = value

    # 3. Create and validate config object
    # This will use defaults for any missing fields
    return CodeFlowConfig(**config_data)
