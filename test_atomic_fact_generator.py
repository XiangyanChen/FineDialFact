"""
Basic tests for AtomicFactGenerator and SQLite3CacheManager.

These tests validate the core functionality of the caching system.
"""

import unittest
import tempfile
import os
import time
from sqlite_cache_manager import SQLite3CacheManager
from atomic_fact_generator import AtomicFactGenerator, MockFactExtractor


class TestSQLite3CacheManager(unittest.TestCase):
    """Test cases for SQLite3CacheManager."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.cache_manager = SQLite3CacheManager(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database."""
        self.cache_manager.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        key1 = self.cache_manager._generate_cache_key("gpt-4", "test prompt")
        key2 = self.cache_manager._generate_cache_key("gpt-4", "test prompt")
        key3 = self.cache_manager._generate_cache_key("gpt-3.5", "test prompt")
        key4 = self.cache_manager._generate_cache_key("gpt-4", "different prompt")
        
        # Same inputs should produce same key
        self.assertEqual(key1, key2)
        
        # Different inputs should produce different keys
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key1, key4)
        
        # Keys should be consistent length (SHA256 hex)
        self.assertEqual(len(key1), 64)
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving cached results."""
        model_name = "gpt-4"
        prompt = "Test prompt for caching"
        atomic_facts = ["Fact 1", "Fact 2", "Fact 3"]
        
        # Store result
        success = self.cache_manager.store_result(model_name, prompt, atomic_facts)
        self.assertTrue(success)
        
        # Retrieve result
        retrieved_facts = self.cache_manager.get_cached_result(model_name, prompt)
        self.assertEqual(retrieved_facts, atomic_facts)
    
    def test_cache_miss(self):
        """Test cache miss scenario."""
        result = self.cache_manager.get_cached_result("unknown-model", "unknown prompt")
        self.assertIsNone(result)
    
    def test_cache_replacement(self):
        """Test cache entry replacement."""
        model_name = "gpt-4"
        prompt = "Test prompt"
        original_facts = ["Original fact 1", "Original fact 2"]
        updated_facts = ["Updated fact 1", "Updated fact 2", "Updated fact 3"]
        
        # Store original
        self.cache_manager.store_result(model_name, prompt, original_facts)
        retrieved = self.cache_manager.get_cached_result(model_name, prompt)
        self.assertEqual(retrieved, original_facts)
        
        # Replace with updated
        self.cache_manager.store_result(model_name, prompt, updated_facts)
        retrieved = self.cache_manager.get_cached_result(model_name, prompt)
        self.assertEqual(retrieved, updated_facts)
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Initially empty
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats['total_entries'], 0)
        
        # Add some entries
        self.cache_manager.store_result("gpt-4", "prompt1", ["fact1"])
        self.cache_manager.store_result("gpt-3.5", "prompt2", ["fact2"])
        self.cache_manager.store_result("gpt-4", "prompt3", ["fact3"])
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats['total_entries'], 3)
        self.assertEqual(stats['unique_models'], 2)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add test data
        self.cache_manager.store_result("gpt-4", "prompt1", ["fact1"])
        self.cache_manager.store_result("gpt-3.5", "prompt2", ["fact2"])
        self.cache_manager.store_result("gpt-4", "prompt3", ["fact3"])
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats['total_entries'], 3)
        
        # Clear specific model
        success = self.cache_manager.clear_cache("gpt-4")
        self.assertTrue(success)
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats['total_entries'], 1)  # Only gpt-3.5 entry remains
        
        # Clear all
        success = self.cache_manager.clear_cache()
        self.assertTrue(success)
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats['total_entries'], 0)


class TestAtomicFactGenerator(unittest.TestCase):
    """Test cases for AtomicFactGenerator."""
    
    def setUp(self):
        """Set up test generator."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Use mock extractor for consistent testing
        self.mock_extractor = MockFactExtractor()
        self.generator = AtomicFactGenerator(
            cache_db_path=self.temp_db.name,
            fact_extractor=self.mock_extractor.extract_facts
        )
    
    def tearDown(self):
        """Clean up test generator."""
        self.generator.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_single_processing(self):
        """Test single prompt processing."""
        model_name = "gpt-4"
        prompt = "The sky is blue. Grass is green."
        
        facts = self.generator.process_single(model_name, prompt)
        
        self.assertIsInstance(facts, list)
        self.assertTrue(len(facts) > 0)
        self.assertTrue(all(isinstance(fact, str) for fact in facts))
    
    def test_caching_behavior(self):
        """Test that caching works correctly."""
        model_name = "gpt-4"
        prompt = "Test prompt for caching behavior"
        
        # First call - cache miss
        start_time = time.time()
        facts1 = self.generator.process_single(model_name, prompt)
        first_duration = time.time() - start_time
        
        # Second call - cache hit
        start_time = time.time()
        facts2 = self.generator.process_single(model_name, prompt)
        second_duration = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(facts1, facts2)
        
        # Second call should be faster (cache hit)
        self.assertLess(second_duration, first_duration)
        
        # Check statistics
        stats = self.generator.get_statistics()
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 1)
        self.assertAlmostEqual(stats['cache_hit_rate'], 0.5, places=2)
    
    def test_force_refresh(self):
        """Test force refresh functionality."""
        model_name = "gpt-4"
        prompt = "Test prompt for force refresh"
        
        # First call
        facts1 = self.generator.process_single(model_name, prompt)
        
        # Second call with force refresh
        facts2 = self.generator.process_single(model_name, prompt, force_refresh=True)
        
        # Results should be identical (same inputs)
        self.assertEqual(facts1, facts2)
        
        # But should have 2 cache misses (no cache hits due to force refresh)
        stats = self.generator.get_statistics()
        self.assertEqual(stats['cache_hits'], 0)
        self.assertEqual(stats['cache_misses'], 2)
    
    def test_batch_processing(self):
        """Test batch processing."""
        requests = [
            {'model_name': 'gpt-4', 'prompt': 'First prompt'},
            {'model_name': 'gpt-3.5-turbo', 'prompt': 'Second prompt'},
            {'model_name': 'gpt-4', 'prompt': 'First prompt'},  # Duplicate
        ]
        
        results = self.generator.process_batch(requests)
        
        self.assertEqual(len(results), 3)
        
        # Check result structure
        for i, result in enumerate(results):
            self.assertIn('model_name', result)
            self.assertIn('prompt', result)
            self.assertIn('atomic_facts', result)
            self.assertIn('fact_count', result)
            self.assertIn('request_index', result)
            self.assertEqual(result['request_index'], i)
        
        # Third request should be a cache hit
        stats = self.generator.get_statistics()
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 2)
    
    def test_invalid_batch_request(self):
        """Test handling of invalid batch requests."""
        requests = [
            {'model_name': 'gpt-4', 'prompt': 'Valid request'},
            {'model_name': 'gpt-4'},  # Missing prompt
            {'prompt': 'Missing model name'},  # Missing model_name
            {'model_name': 'gpt-4', 'prompt': 'Another valid request'},
        ]
        
        results = self.generator.process_batch(requests)
        
        # Should still process valid requests
        valid_results = [r for r in results if 'error' not in r]
        self.assertEqual(len(valid_results), 2)
    
    def test_cache_disabled(self):
        """Test generator with caching disabled."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        try:
            generator = AtomicFactGenerator(
                cache_db_path=temp_db.name,
                enable_cache=False,
                fact_extractor=self.mock_extractor.extract_facts
            )
            
            model_name = "gpt-4"
            prompt = "Test prompt without caching"
            
            # Process twice
            facts1 = generator.process_single(model_name, prompt)
            facts2 = generator.process_single(model_name, prompt)
            
            # Results should be identical
            self.assertEqual(facts1, facts2)
            
            # But should have 2 cache misses (cache disabled)
            stats = generator.get_statistics()
            self.assertEqual(stats['cache_hits'], 0)
            self.assertEqual(stats['cache_misses'], 2)
            self.assertEqual(stats['cache_hit_rate'], 0.0)
            
            generator.close()
            
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    def test_cache_management(self):
        """Test cache management operations."""
        # Add test data
        self.generator.process_single("gpt-4", "prompt1")
        self.generator.process_single("gpt-3.5-turbo", "prompt2")
        self.generator.process_single("gpt-4", "prompt3")
        
        stats = self.generator.get_statistics()
        initial_entries = stats.get('cache_total_entries', 0)
        self.assertGreater(initial_entries, 0)
        
        # Clear specific model
        success = self.generator.clear_cache("gpt-4")
        self.assertTrue(success)
        
        # Clear all
        success = self.generator.clear_cache()
        self.assertTrue(success)
        
        stats = self.generator.get_statistics()
        final_entries = stats.get('cache_total_entries', 0)
        self.assertEqual(final_entries, 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Run tests
    unittest.main(verbosity=2)