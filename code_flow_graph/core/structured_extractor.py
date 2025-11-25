import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
import time

from .models import StructuredDataElement
from .utils import (
    get_gitignore_patterns,
    match_file_against_pattern
)

class StructuredDataExtractor:
    """
    Extractor for structured data files (JSON, YAML).
    Parses files and generates semantic chunks based on the data hierarchy.
    """

    def __init__(self, ignored_filenames: Optional[Set[str]] = None):
        self.ignored_filenames = ignored_filenames or {
            'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 
            'composer.lock', 'poetry.lock', 'Gemfile.lock', 'cargo.lock'
        }
        self.supported_extensions = {'.json', '.yaml', '.yml'}

    def extract_from_directory(self, directory: Path) -> List[StructuredDataElement]:
        """
        Extract structured data elements from all supported files in a directory.
        Respects .gitignore patterns and ignored filenames.
        """
        elements = []
        directory = directory.resolve()
        
        # Gather all potentially relevant files
        all_files = []
        for ext in self.supported_extensions:
            all_files.extend(directory.rglob(f"*{ext}"))
            
        # Get gitignore patterns
        ignored_patterns_with_dirs = get_gitignore_patterns(directory)
        
        filtered_files = []
        for file_path in all_files:
            # Check ignored filenames
            if file_path.name in self.ignored_filenames:
                continue
                
            # Check gitignore
            if any(match_file_against_pattern(file_path, pattern, gitignore_dir, directory)
                   for pattern, gitignore_dir in ignored_patterns_with_dirs):
                continue
                
            filtered_files.append(file_path)
            
        logging.info(f"Found {len(filtered_files)} structured data files to analyze.")
        
        for file_path in filtered_files:
            try:
                elements.extend(self.extract_from_file(file_path))
            except Exception as e:
                logging.warning(f"Failed to extract from {file_path}: {e}")
                
        return elements

    def extract_from_file(self, file_path: Path) -> List[StructuredDataElement]:
        """
        Extract elements from a single structured data file.
        """
        if not file_path.exists():
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            data = None
            if file_path.suffix == '.json':
                data = json.loads(content)
            elif file_path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(content)
                
            if data is None:
                return []
                
            return self._chunk_data(data, file_path, content)
            
        except Exception as e:
            logging.warning(f"Error processing {file_path}: {e}")
            return []

    def _chunk_data(self, data: Any, file_path: Path, full_source: str) -> List[StructuredDataElement]:
        """
        Flatten the data structure into semantic chunks.
        """
        chunks = []
        
        def _traverse(current_data: Any, path: str, key: str):
            # Create a chunk for the current node
            value_type = type(current_data).__name__
            
            # Determine content representation
            if isinstance(current_data, (dict, list)):
                # For containers, create a summary chunk
                if isinstance(current_data, dict):
                    keys = list(current_data.keys())[:10] # Limit keys in summary
                    content = f"{path}: Object with keys {keys}"
                    if len(current_data) > 10:
                        content += "..."
                else:
                    content = f"{path}: Array with {len(current_data)} items"
            else:
                # For leaf nodes, show the value
                content = f"{path}: {current_data}"
            
            # Create element
            # Note: Line numbers are hard to get with standard json/yaml parsers without custom loaders.
            # We'll default to 1 for now, or 0.
            chunk = StructuredDataElement(
                name=key,
                kind='structured_data',
                file_path=str(file_path),
                line_start=1, 
                line_end=1,
                full_source=full_source,
                json_path=path,
                value_type=value_type,
                key_name=key,
                content=content,
                metadata={
                    'file_type': file_path.suffix[1:], # json or yaml
                    'json_path': path
                }
            )
            chunks.append(chunk)
            
            # Recurse
            if isinstance(current_data, dict):
                for k, v in current_data.items():
                    new_path = f"{path}.{k}" if path else k
                    _traverse(v, new_path, k)
            elif isinstance(current_data, list):
                for i, item in enumerate(current_data):
                    new_path = f"{path}[{i}]"
                    _traverse(item, new_path, str(i))

        # Start traversal
        _traverse(data, "", "root")
        
        return chunks
