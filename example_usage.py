"""
Example usage of the AtomicFactGenerator with SQLite3 caching.

This script demonstrates how to use the AtomicFactGenerator with caching
to process text prompts and extract atomic facts efficiently.
"""

import logging
import time
from atomic_fact_generator import AtomicFactGenerator, MockFactExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demonstrate_basic_usage():
    """Demonstrate basic usage of AtomicFactGenerator."""
    print("=== Basic Usage Demonstration ===")
    
    # Initialize the generator with caching enabled
    generator = AtomicFactGenerator(cache_db_path="demo_cache.db")
    
    # Sample prompts for testing
    sample_prompts = [
        {
            'model_name': 'gpt-4',
            'prompt': 'The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum. The city has a population of over 2 million people.'
        },
        {
            'model_name': 'gpt-3.5-turbo',
            'prompt': 'Python is a programming language. It was created by Guido van Rossum and it supports multiple programming paradigms.'
        },
        {
            'model_name': 'claude-2',
            'prompt': 'Machine learning is a subset of artificial intelligence. It enables computers to learn without being explicitly programmed.'
        }
    ]
    
    # Process prompts individually
    print("\n--- Processing Individual Prompts ---")
    for prompt_data in sample_prompts:
        print(f"\nProcessing with {prompt_data['model_name']}:")
        print(f"Prompt: {prompt_data['prompt'][:50]}...")
        
        facts = generator.process_single(
            prompt_data['model_name'],
            prompt_data['prompt']
        )
        
        print(f"Extracted {len(facts)} atomic facts:")
        for i, fact in enumerate(facts, 1):
            print(f"  {i}. {fact}")
    
    # Show statistics
    print(f"\nStatistics: {generator.get_statistics()}")
    
    generator.close()

def demonstrate_caching():
    """Demonstrate caching functionality."""
    print("\n\n=== Caching Demonstration ===")
    
    # Initialize generator with mock fact extractor for consistent timing
    mock_extractor = MockFactExtractor()
    generator = AtomicFactGenerator(
        cache_db_path="caching_demo.db",
        fact_extractor=mock_extractor.extract_facts
    )
    
    test_prompt = {
        'model_name': 'gpt-4',
        'prompt': 'The Earth is the third planet from the Sun. It has one moon and supports life through its atmosphere and water.'
    }
    
    print("\n--- First Processing (Cache Miss) ---")
    start_time = time.time()
    facts1 = generator.process_single(test_prompt['model_name'], test_prompt['prompt'])
    first_time = time.time() - start_time
    print(f"Processing time: {first_time:.2f}s")
    print(f"Extracted facts: {facts1}")
    
    print("\n--- Second Processing (Cache Hit) ---")
    start_time = time.time()
    facts2 = generator.process_single(test_prompt['model_name'], test_prompt['prompt'])
    second_time = time.time() - start_time
    print(f"Processing time: {second_time:.2f}s")
    print(f"Extracted facts: {facts2}")
    
    print(f"\nTime saved: {first_time - second_time:.2f}s")
    print(f"Facts are identical: {facts1 == facts2}")
    
    # Show cache statistics
    stats = generator.get_statistics()
    print(f"\nCache statistics:")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Time saved: {stats['processing_time_saved']:.2f}s")
    
    generator.close()

def demonstrate_batch_processing():
    """Demonstrate batch processing with mixed cache hits and misses."""
    print("\n\n=== Batch Processing Demonstration ===")
    
    mock_extractor = MockFactExtractor()
    generator = AtomicFactGenerator(
        cache_db_path="batch_demo.db",
        fact_extractor=mock_extractor.extract_facts
    )
    
    # Batch of requests with some duplicates to show caching
    batch_requests = [
        {
            'model_name': 'gpt-4',
            'prompt': 'Water boils at 100 degrees Celsius. It freezes at 0 degrees Celsius.'
        },
        {
            'model_name': 'gpt-3.5-turbo',
            'prompt': 'The human brain has billions of neurons. It controls all bodily functions.'
        },
        {
            'model_name': 'claude-2',
            'prompt': 'Solar energy is renewable. It comes from the sun and can power homes.'
        },
        {
            'model_name': 'gpt-4',  # Same model, same prompt - should hit cache
            'prompt': 'Water boils at 100 degrees Celsius. It freezes at 0 degrees Celsius.'
        },
        {
            'model_name': 'llama-2',
            'prompt': 'Photosynthesis converts sunlight into energy. Plants use this process to grow.'
        }
    ]
    
    print(f"\nProcessing batch of {len(batch_requests)} requests...")
    start_time = time.time()
    results = generator.process_batch(batch_requests)
    total_time = time.time() - start_time
    
    print(f"Batch processing completed in {total_time:.2f}s")
    print(f"Processed {len(results)} requests")
    
    # Show results
    for i, result in enumerate(results):
        print(f"\nRequest {i + 1}:")
        print(f"  Model: {result['model_name']}")
        print(f"  Facts: {result['fact_count']}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            for fact in result['atomic_facts'][:2]:  # Show first 2 facts
                print(f"    - {fact}")
            if result['fact_count'] > 2:
                print(f"    ... and {result['fact_count'] - 2} more")
    
    # Final statistics
    stats = generator.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total requests: {stats['cache_hits'] + stats['cache_misses']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Processing time saved: {stats['processing_time_saved']:.2f}s")
    
    if 'cache_total_entries' in stats:
        print(f"  Cache entries: {stats['cache_total_entries']}")
        print(f"  Unique models: {stats['cache_unique_models']}")
    
    generator.close()

def demonstrate_cache_management():
    """Demonstrate cache management operations."""
    print("\n\n=== Cache Management Demonstration ===")
    
    generator = AtomicFactGenerator(cache_db_path="management_demo.db")
    
    # Add some test data
    test_data = [
        {'model_name': 'gpt-4', 'prompt': 'Test prompt 1'},
        {'model_name': 'gpt-3.5-turbo', 'prompt': 'Test prompt 2'},
        {'model_name': 'gpt-4', 'prompt': 'Test prompt 3'},
    ]
    
    print("Adding test data to cache...")
    for data in test_data:
        generator.process_single(data['model_name'], data['prompt'])
    
    # Show cache stats
    stats = generator.get_statistics()
    print(f"Cache entries before clearing: {stats.get('cache_total_entries', 'Unknown')}")
    
    # Clear cache for specific model
    print(f"\nClearing cache for 'gpt-4'...")
    generator.clear_cache('gpt-4')
    
    stats = generator.get_statistics()
    print(f"Cache entries after clearing gpt-4: {stats.get('cache_total_entries', 'Unknown')}")
    
    # Clear all cache
    print(f"\nClearing entire cache...")
    generator.clear_cache()
    
    stats = generator.get_statistics()
    print(f"Cache entries after clearing all: {stats.get('cache_total_entries', 'Unknown')}")
    
    generator.close()

if __name__ == "__main__":
    try:
        demonstrate_basic_usage()
        demonstrate_caching()
        demonstrate_batch_processing()
        demonstrate_cache_management()
        
        print("\n\n=== Demo Completed Successfully ===")
        print("Check the generated .db files to see the SQLite cache databases.")
        
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        raise