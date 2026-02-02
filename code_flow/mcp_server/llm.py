import asyncio
import logging
import os
from typing import Optional, List
import json

from openai import AsyncOpenAI, DefaultAioHttpClient
from code_flow.core.call_graph_builder import FunctionNode
from code_flow.core.vector_store import CodeVectorStore

logger = logging.getLogger("mcp.llm")

class SummaryGenerator:
    """Generates natural language summaries for code elements using an LLM."""

    def __init__(self, config: dict):
        self.config = config.get('llm_config', {})
        self.api_key = os.environ.get(self.config.get('api_key_env_var', 'OPENAI_API_KEY'))
        
        # Priority: Env Var > Config > Default
        self.base_url = os.environ.get('OPENAI_BASE_URL', self.config.get('base_url', 'https://openrouter.ai/api/v1'))
        self.model = os.environ.get('OPENAI_SUMMARY_MODEL', self.config.get('model', 'x-ai/grok-4.1-fast'))
        
        self.max_tokens = self.config.get('max_tokens', 256)
        self.max_input_tokens = self.config.get('max_input_tokens', 2000)
        self.summary_depth = self.config.get('summary_depth', 'standard')
        
        # Smart filtering options
        self.min_complexity = self.config.get('min_complexity', 0)
        self.min_nloc = self.config.get('min_nloc', 0)
        self.skip_private = self.config.get('skip_private', False)
        self.skip_test = self.config.get('skip_test', False)
        
        if not self.api_key:
            logger.warning("No API key found for SummaryGenerator. Summarization will fail.")

        # Initialize OpenAI client with aiohttp for better concurrency
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=DefaultAioHttpClient()
        )

    def should_summarize(self, node: FunctionNode) -> bool:
        """Determine if a function should be summarized based on filtering rules."""
        # Check complexity threshold
        if node.complexity is not None and node.complexity < self.min_complexity:
            return False
        
        # Check NLOC threshold
        if node.nloc is not None and node.nloc < self.min_nloc:
            return False
        
        # Skip private functions if configured
        if self.skip_private:
            # Python convention: names starting with _ (but not __)
            if node.name.startswith('_') and not node.name.startswith('__'):
                return False
            
            # TypeScript/Java convention: access_modifier field
            if node.access_modifier in ['private', 'protected']:
                return False
        
        # Skip test functions if configured
        if self.skip_test:
            name_lower = node.name.lower()
            if name_lower.startswith('test_') or name_lower.endswith('_test') or 'test' in name_lower:
                return False
        
        return True

    async def generate_summary(self, node: FunctionNode, source_code: str) -> Optional[str]:
        """
        Generate a concise summary for a function node.
        
        Args:
            node: The FunctionNode to summarize.
            source_code: The source code of the function.
            
        Returns:
            The generated summary string, or None if generation failed.
        """
        if not self.api_key:
            return None
        
        # Check if this function should be summarized
        if not self.should_summarize(node):
            logger.debug(f"Skipping {node.fully_qualified_name} based on filtering rules")
            return None

        try:
            prompt = self._construct_prompt(node, source_code)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert software engineer. Generate a concise, natural language summary for the provided code component. Focus on WHAT it does and WHY, not just translating code to text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary for {node.fully_qualified_name}: {e}")
            return None

    def _construct_prompt(self, node: FunctionNode, source_code: str) -> str:
        """Construct the prompt for the LLM based on depth setting."""
        signature = f"def {node.name}({', '.join(node.parameters)}) -> {node.return_type}"
        
        # Truncate source code if too long
        if len(source_code) > self.max_input_tokens:
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = self.max_input_tokens * 4
            source_code = source_code[:max_chars] + "\n... (truncated)"
        
        if self.summary_depth == "minimal":
            # Minimal: Just name, signature, and code
            return f"""
Component Type: {'Method' if node.is_method else 'Function'}
Name: {node.name}
Signature: {signature}

Code:
{source_code}

Summary:
"""
        elif self.summary_depth == "detailed":
            # Detailed: Include all available metadata
            docstring = node.docstring or "None"
            complexity = f"Complexity: {node.complexity}" if node.complexity else ""
            nloc = f"NLOC: {node.nloc}" if node.nloc else ""
            decorators = f"Decorators: {[d.get('name', '') for d in node.decorators]}" if node.decorators else ""
            
            return f"""
Component Type: {'Method' if node.is_method else 'Function'}
Name: {node.name}
Signature: {signature}
Docstring: {docstring}
{complexity}
{nloc}
{decorators}

Code:
{source_code}

Summary:
"""
        else:
            # Standard: Name, signature, docstring, code
            docstring = node.docstring or "None"
            return f"""
Component Type: {'Method' if node.is_method else 'Function'}
Name: {node.name}
Signature: {signature}
Docstring: {docstring}

Code:
{source_code}

Summary:
"""

class SummaryProcessor:
    """Manages background processing of summary generation."""

    def __init__(self, generator: SummaryGenerator, builder, vector_store: CodeVectorStore, concurrency: int = 5, prioritize_entry_points: bool = False):
        self.generator = generator
        self.builder = builder
        self.vector_store = vector_store
        self.queue = asyncio.Queue()
        self.concurrency = concurrency
        self.workers = []
        self.running = False
        self.prioritize_entry_points = prioritize_entry_points

    def start(self):
        """Start the background worker tasks."""
        if self.running:
            return
        
        self.running = True
        
        # Check if we're in an async context or need to create one
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context (MCP server)
            for _ in range(self.concurrency):
                task = asyncio.create_task(self._worker())
                self.workers.append(task)
        except RuntimeError:
            # No running loop (CLI tool) - we'll create tasks when needed
            # Workers will be created in the event loop when we call asyncio.run()
            pass
        
        logger.info(f"Started {self.concurrency} summary processor workers")

    async def start_async(self):
        """Start workers in an async context."""
        if self.running:
            return
        
        self.running = True
        for _ in range(self.concurrency):
            task = asyncio.create_task(self._worker())
            self.workers.append(task)
        
        logger.info(f"Started {self.concurrency} summary processor workers")

    async def stop(self):
        """Stop the background workers."""
        self.running = False
        # Send sentinel values to stop workers
        for _ in range(self.concurrency):
            await self.queue.put(None)
        
        if self.workers:
            await asyncio.gather(*self.workers)
        
        logger.info("Stopped summary processor workers")

    def enqueue(self, fqn: str):
        """Enqueue a function FQN for summarization."""
        self.queue.put_nowait(fqn)

    async def _worker(self):
        """Worker task to process the queue."""
        while self.running:
            try:
                fqn = await self.queue.get()
                
                if fqn is None:
                    self.queue.task_done()
                    break

                try:
                    # Get node from builder
                    node = self.builder.functions.get(fqn)
                    if not node:
                        logger.warning(f"Node {fqn} not found in builder, skipping summary")
                        continue

                    # Get source code (we need to read file again or have it cached)
                    # For now, read file. Optimization: Cache source in builder or pass it.
                    # Reading file is safer for latest content.
                    try:
                        with open(node.file_path, 'r', encoding='utf-8') as f:
                            full_source = f.read()
                            lines = full_source.split('\n')
                            start = max(0, node.line_start - 1)
                            end = node.line_end
                            func_source = '\n'.join(lines[start:end])
                    except Exception as e:
                        logger.warning(f"Could not read source for {fqn}: {e}")
                        continue

                    # Generate summary
                    summary = await self.generator.generate_summary(node, func_source)
                    
                    if summary:
                        # Update node
                        node.summary = summary
                        
                        # Update vector store
                        # We need to pass full source to update_function_node as it re-chunks
                        self.vector_store.update_function_node(node, full_source)
                        logger.info(f"Generated and stored summary for {fqn}")
                    
                except Exception as e:
                    logger.error(f"Error processing summary for {fqn}: {e}")
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
