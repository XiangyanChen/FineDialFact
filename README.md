# AtomicFactGenerator with SQLite3 Caching

A high-performance atomic fact generation system with SQLite3-based caching to avoid redundant processing of identical inputs.

## Features

- **SQLite3 Caching**: Efficient database-backed caching system
- **Cache Key Strategy**: Hash-based keys using model name + prompt combinations
- **Thread-Safe Operations**: Proper locking for concurrent access
- **Batch Processing**: Process multiple requests efficiently
- **Cache Management**: Clear specific models or entire cache
- **Statistics Tracking**: Monitor cache performance and processing metrics
- **Error Handling**: Robust error management and database connection handling
- **Zero External Dependencies**: Uses only Python standard library

## Quick Start

```python
from atomic_fact_generator import AtomicFactGenerator

# Initialize with caching enabled
generator = AtomicFactGenerator(cache_db_path="facts_cache.db")

# Process a single prompt
facts = generator.process_single(
    model_name="gpt-4",
    prompt="The Earth orbits the Sun. It has one moon."
)

print(f"Extracted {len(facts)} atomic facts:")
for fact in facts:
    print(f"- {fact}")

# Process multiple prompts
requests = [
    {"model_name": "gpt-4", "prompt": "Python is a programming language."},
    {"model_name": "gpt-3.5-turbo", "prompt": "Water boils at 100°C."}
]

results = generator.process_batch(requests)

# View statistics
stats = generator.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")

# Clean up
generator.close()
```

## Core Components

### SQLite3CacheManager

The database wrapper class that handles all cache operations:

- **Cache Storage**: Stores atomic facts with model name and prompt hash
- **Key Generation**: Creates unique SHA256 keys from model + prompt combinations
- **Database Management**: Handles connections, transactions, and schema
- **Statistics**: Provides cache metrics and database information

### AtomicFactGenerator

The main processing class with integrated caching:

- **Fact Extraction**: Processes text to extract atomic facts
- **Cache Integration**: Automatically checks cache before processing
- **Batch Processing**: Efficiently handles multiple requests
- **Custom Extractors**: Supports pluggable fact extraction functions

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   User Input        │    │ AtomicFactGenerator │    │ SQLite3CacheManager │
│                     │    │                     │    │                     │
│ model_name + prompt │───▶│ 1. Check cache      │───▶│ get_cached_result() │
│                     │    │ 2. Process if miss  │    │                     │
│                     │    │ 3. Store result     │───▶│ store_result()      │
│                     │    │ 4. Return facts     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Testing

Run the included tests to validate functionality:

```bash
python test_atomic_fact_generator.py
```

Run the example demonstration:

```bash
python example_usage.py
```

## File Structure

```
.
├── atomic_fact_generator.py     # Main generator class
├── sqlite_cache_manager.py      # Database wrapper
├── example_usage.py            # Usage demonstrations
├── test_atomic_fact_generator.py # Unit tests
├── requirements.txt            # Dependencies (standard library only)
└── README.md                  # This documentation
```

This implementation is designed to be easily integrated into any project with minimal dependencies and maximum performance.