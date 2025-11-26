import os
import yaml
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

# Default configuration values
DEFAULT_CONFIG_PATH = "codeflow.config.yaml"
DEFAULT_WATCH_DIRECTORIES = ["."]
DEFAULT_IGNORED_PATTERNS = ["venv", "**/__pycache__", ".git", ".idea", ".vscode", "node_modules"]
DEFAULT_CHROMADB_PATH = "./code_vectors_chroma"
DEFAULT_MAX_GRAPH_DEPTH = 3
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MAX_TOKENS = 256
DEFAULT_LANGUAGE = "python"


class CodeFlowConfig(BaseModel):
    """
    Central configuration model for CodeFlowGraph.
    """
    watch_directories: List[str] = Field(default_factory=lambda: DEFAULT_WATCH_DIRECTORIES)
    ignored_patterns: List[str] = Field(default_factory=lambda: DEFAULT_IGNORED_PATTERNS)
    chromadb_path: str = Field(default=DEFAULT_CHROMADB_PATH)
    max_graph_depth: int = Field(default=DEFAULT_MAX_GRAPH_DEPTH)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS)
    language: str = Field(default=DEFAULT_LANGUAGE)
    
    # Optional LLM config for summaries (can be expanded)
    summary_generation_enabled: bool = False
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Allow extra fields for flexibility
    class Config:
        extra = "allow"


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
