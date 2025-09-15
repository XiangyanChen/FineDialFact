"""
AtomicFactGenerator with SQLite3 caching support.

This module provides the main AtomicFactGenerator class that processes responses
to extract atomic facts, with built-in caching to avoid redundant processing.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from sqlite_cache_manager import SQLite3CacheManager


class AtomicFactGenerator:
    """
    Generator for extracting atomic facts from text responses with SQLite3 caching.
    
    This class processes responses to extract atomic facts and caches results
    based on the combination of model name and prompt to avoid redundant processing.
    """
    
    def __init__(self, 
                 cache_db_path: str = "atomic_facts_cache.db",
                 enable_cache: bool = True,
                 fact_extractor: Optional[Callable[[str, str], List[str]]] = None):
        """
        Initialize the AtomicFactGenerator.
        
        Args:
            cache_db_path: Path to the SQLite cache database.
            enable_cache: Whether to enable caching functionality.
            fact_extractor: Custom function for extracting facts from text.
                           Should take (model_name, prompt) and return List[str].
                           If None, uses the default implementation.
        """
        self.enable_cache = enable_cache
        self.cache_manager = SQLite3CacheManager(cache_db_path) if enable_cache else None
        self.fact_extractor = fact_extractor or self._default_fact_extractor
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_time_saved': 0.0,
            'total_processed': 0
        }
    
    def _default_fact_extractor(self, model_name: str, prompt: str) -> List[str]:
        """
        Default implementation for extracting atomic facts.
        
        This is a placeholder implementation that simulates fact extraction.
        In a real implementation, this would involve calling an LLM or
        using sophisticated NLP techniques.
        
        Args:
            model_name: Name of the model to use for fact extraction.
            prompt: The text prompt to extract facts from.
            
        Returns:
            List of extracted atomic facts.
        """
        self.logger.debug(f"Extracting facts using model: {model_name}")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Simple fact extraction simulation
        sentences = prompt.split('.')
        facts = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Basic filtering
                # Split compound sentences
                if ' and ' in sentence:
                    parts = sentence.split(' and ')
                    facts.extend([part.strip() for part in parts if part.strip()])
                elif ' but ' in sentence:
                    parts = sentence.split(' but ')
                    facts.extend([part.strip() for part in parts if part.strip()])
                else:
                    facts.append(sentence)
        
        # Remove duplicates while preserving order
        unique_facts = []
        seen = set()
        for fact in facts:
            if fact not in seen:
                unique_facts.append(fact)
                seen.add(fact)
        
        self.logger.debug(f"Extracted {len(unique_facts)} atomic facts")
        return unique_facts
    
    def process_single(self, model_name: str, prompt: str, force_refresh: bool = False) -> List[str]:
        """
        Process a single prompt to extract atomic facts.
        
        Args:
            model_name: Name of the model to use for processing.
            prompt: The text prompt to process.
            force_refresh: If True, skip cache and force new processing.
            
        Returns:
            List of atomic facts extracted from the prompt.
        """
        start_time = time.time()
        
        # Check cache first (unless forced refresh)
        if self.enable_cache and not force_refresh:
            cached_result = self.cache_manager.get_cached_result(model_name, prompt)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                self.stats['processing_time_saved'] += time.time() - start_time
                self.logger.info(f"Cache hit for model: {model_name}")
                return cached_result
        
        # Cache miss or cache disabled - process normally
        self.stats['cache_misses'] += 1
        self.logger.info(f"Processing new request for model: {model_name}")
        
        try:
            atomic_facts = self.fact_extractor(model_name, prompt)
            
            # Store result in cache
            if self.enable_cache:
                success = self.cache_manager.store_result(model_name, prompt, atomic_facts)
                if not success:
                    self.logger.warning("Failed to store result in cache")
            
            self.stats['total_processed'] += 1
            processing_time = time.time() - start_time
            self.logger.info(f"Processed in {processing_time:.2f}s, extracted {len(atomic_facts)} facts")
            
            return atomic_facts
            
        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}")
            raise
    
    def process_batch(self, requests: List[Dict[str, str]], force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Process a batch of requests to extract atomic facts.
        
        Args:
            requests: List of dictionaries with 'model_name' and 'prompt' keys.
            force_refresh: If True, skip cache and force new processing.
            
        Returns:
            List of dictionaries containing input info and extracted atomic facts.
        """
        results = []
        
        self.logger.info(f"Processing batch of {len(requests)} requests")
        
        for i, request in enumerate(requests):
            if 'model_name' not in request or 'prompt' not in request:
                self.logger.error(f"Invalid request format at index {i}: {request}")
                continue
            
            try:
                atomic_facts = self.process_single(
                    request['model_name'],
                    request['prompt'],
                    force_refresh
                )
                
                result = {
                    'model_name': request['model_name'],
                    'prompt': request['prompt'],
                    'atomic_facts': atomic_facts,
                    'fact_count': len(atomic_facts),
                    'request_index': i
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing request {i}: {e}")
                # Add error result to maintain consistency
                results.append({
                    'model_name': request['model_name'],
                    'prompt': request['prompt'],
                    'atomic_facts': [],
                    'fact_count': 0,
                    'request_index': i,
                    'error': str(e)
                })
        
        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing and cache statistics.
        
        Returns:
            Dictionary containing various statistics.
        """
        stats = self.stats.copy()
        
        # Calculate additional metrics
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        # Add cache statistics if available
        if self.enable_cache and self.cache_manager:
            cache_stats = self.cache_manager.get_cache_stats()
            stats.update({f"cache_{k}": v for k, v in cache_stats.items()})
        
        return stats
    
    def clear_cache(self, model_name: Optional[str] = None) -> bool:
        """
        Clear the cache.
        
        Args:
            model_name: If provided, only clear entries for this model.
                       If None, clear all entries.
        
        Returns:
            True if cleared successfully, False otherwise.
        """
        if not self.enable_cache:
            self.logger.warning("Cache is disabled, nothing to clear")
            return False
        
        return self.cache_manager.clear_cache(model_name)
    
    def close(self):
        """Close the generator and clean up resources."""
        if self.cache_manager:
            self.cache_manager.close()
        self.logger.info("AtomicFactGenerator closed")


class MockFactExtractor:
    """
    Mock fact extractor for testing and demonstration purposes.
    
    This class simulates different models with varying processing characteristics.
    """
    
    def __init__(self):
        self.models = {
            'gpt-3.5-turbo': {'delay': 0.5, 'quality': 'good'},
            'gpt-4': {'delay': 1.0, 'quality': 'excellent'},
            'claude-2': {'delay': 0.8, 'quality': 'excellent'},
            'llama-2': {'delay': 0.3, 'quality': 'fair'}
        }
    
    def extract_facts(self, model_name: str, prompt: str) -> List[str]:
        """
        Mock fact extraction with simulated model behavior.
        
        Args:
            model_name: Name of the model to simulate.
            prompt: The text prompt to process.
            
        Returns:
            List of simulated atomic facts.
        """
        model_config = self.models.get(model_name, {'delay': 0.2, 'quality': 'basic'})
        
        # Simulate model processing time
        time.sleep(model_config['delay'])
        
        # Generate facts based on model quality
        base_facts = prompt.split('. ')
        base_facts = [fact.strip() for fact in base_facts if fact.strip()]
        
        if model_config['quality'] == 'excellent':
            # High-quality models extract more detailed facts
            facts = []
            for fact in base_facts:
                facts.append(fact)
                if ' and ' in fact:
                    facts.extend(fact.split(' and '))
        elif model_config['quality'] == 'good':
            # Good models do basic fact separation
            facts = base_facts
        else:
            # Basic models may miss some facts
            facts = base_facts[:max(1, len(base_facts) // 2)]
        
        return [f"[{model_name}] {fact}" for fact in facts if fact]