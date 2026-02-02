import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from code_flow.mcp_server.llm import SummaryGenerator, SummaryProcessor
from code_flow.core.call_graph_builder import FunctionNode

class TestSummaryGeneration(unittest.IsolatedAsyncioTestCase):
    async def test_summary_generator(self):
        config = {
            'llm_config': {
                'api_key_env_var': 'TEST_KEY',
                'model': 'gpt-4o'
            }
        }
        
        with patch.dict('os.environ', {'TEST_KEY': 'sk-test'}):
            generator = SummaryGenerator(config)
            
            # Mock OpenAI client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="This is a summary."))]
            generator.client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            node = FunctionNode(
                name="test_func",
                fully_qualified_name="test_func",
                file_path="test.py",
                line_start=1,
                line_end=5,
                parameters=["a", "b"],
                return_type="int"
            )
            
            summary = await generator.generate_summary(node, "def test_func(a, b): return a + b")
            self.assertEqual(summary, "This is a summary.")

    async def test_summary_processor(self):
        # Mock dependencies
        generator = AsyncMock()
        generator.generate_summary.return_value = "Generated Summary"
        
        builder = MagicMock()
        node = FunctionNode(
            name="test_func",
            fully_qualified_name="test_func",
            file_path="test.py",
            line_start=1,
            line_end=2,
            parameters=[]
        )
        builder.functions = {"test_func": node}
        
        vector_store = MagicMock()
        
        processor = SummaryProcessor(generator, builder, vector_store, concurrency=1)
        
        # Mock file reading
        with patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="def test_func():\n    pass"):
            processor.start()
            processor.enqueue("test_func")
            
            # Wait a bit for processing
            await asyncio.sleep(0.1)
            
            await processor.stop()
            
            # Verify
            generator.generate_summary.assert_called_once()
            self.assertEqual(node.summary, "Generated Summary")
            vector_store.update_function_node.assert_called_once()

if __name__ == '__main__':
    unittest.main()
