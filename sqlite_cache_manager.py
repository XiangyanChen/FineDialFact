"""
SQLite3-based caching system for AtomicFactGenerator.

This module provides a database wrapper class to handle cache operations
for storing and retrieving atomic facts based on model name and prompt combinations.
"""

import sqlite3
import hashlib
import json
import threading
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class SQLite3CacheManager:
    """
    SQLite3 database wrapper for caching atomic facts results.
    
    The cache uses a combination of model name and prompt as the key,
    and stores the atomic facts results as JSON.
    """
    
    def __init__(self, db_path: str = "atomic_facts_cache.db"):
        """
        Initialize the cache manager.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_database()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize the database and create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS atomic_facts_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    atomic_facts TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_key 
                ON atomic_facts_cache(cache_key)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_prompt 
                ON atomic_facts_cache(model_name, prompt_hash)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path, 
                timeout=30.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _generate_cache_key(self, model_name: str, prompt: str) -> str:
        """
        Generate a unique cache key from model name and prompt.
        
        Args:
            model_name: Name of the model used for processing.
            prompt: The prompt text.
            
        Returns:
            A unique hash string representing the cache key.
        """
        combined = f"{model_name}::{prompt}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _generate_prompt_hash(self, prompt: str) -> str:
        """Generate a hash for the prompt for indexing purposes."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def get_cached_result(self, model_name: str, prompt: str) -> Optional[List[str]]:
        """
        Retrieve cached atomic facts for a given model and prompt.
        
        Args:
            model_name: Name of the model.
            prompt: The prompt text.
            
        Returns:
            List of atomic facts if found in cache, None otherwise.
        """
        cache_key = self._generate_cache_key(model_name, prompt)
        
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT atomic_facts FROM atomic_facts_cache WHERE cache_key = ?",
                        (cache_key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        self.logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                        return json.loads(row['atomic_facts'])
                    
                    self.logger.debug(f"Cache miss for key: {cache_key[:16]}...")
                    return None
                    
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.logger.error(f"Error retrieving cached result: {e}")
            return None
    
    def store_result(self, model_name: str, prompt: str, atomic_facts: List[str]) -> bool:
        """
        Store atomic facts in the cache.
        
        Args:
            model_name: Name of the model.
            prompt: The prompt text.
            atomic_facts: List of atomic facts to cache.
            
        Returns:
            True if stored successfully, False otherwise.
        """
        cache_key = self._generate_cache_key(model_name, prompt)
        prompt_hash = self._generate_prompt_hash(prompt)
        
        try:
            with self._lock:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO atomic_facts_cache 
                        (cache_key, model_name, prompt_hash, atomic_facts, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        cache_key,
                        model_name,
                        prompt_hash,
                        json.dumps(atomic_facts)
                    ))
                    conn.commit()
                    
                    self.logger.debug(f"Stored result for key: {cache_key[:16]}...")
                    return True
                    
        except (sqlite3.Error, TypeError) as e:
            self.logger.error(f"Error storing result: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        COUNT(DISTINCT model_name) as unique_models,
                        MIN(created_at) as oldest_entry,
                        MAX(created_at) as newest_entry
                    FROM atomic_facts_cache
                """)
                row = cursor.fetchone()
                
                return {
                    'total_entries': row['total_entries'],
                    'unique_models': row['unique_models'],
                    'oldest_entry': row['oldest_entry'],
                    'newest_entry': row['newest_entry'],
                    'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clear_cache(self, model_name: Optional[str] = None) -> bool:
        """
        Clear cache entries.
        
        Args:
            model_name: If provided, only clear entries for this model.
                       If None, clear all entries.
            
        Returns:
            True if cleared successfully, False otherwise.
        """
        try:
            with self._lock:
                with self._get_connection() as conn:
                    if model_name:
                        cursor = conn.execute(
                            "DELETE FROM atomic_facts_cache WHERE model_name = ?",
                            (model_name,)
                        )
                        deleted_count = cursor.rowcount
                        self.logger.info(f"Cleared {deleted_count} entries for model: {model_name}")
                    else:
                        cursor = conn.execute("DELETE FROM atomic_facts_cache")
                        deleted_count = cursor.rowcount
                        self.logger.info(f"Cleared all {deleted_count} cache entries")
                    
                    conn.commit()
                    return True
                    
        except sqlite3.Error as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
    
    def close(self):
        """Close the cache manager and clean up resources."""
        # SQLite connections are closed automatically via context manager
        # This method is provided for API completeness
        self.logger.debug("Cache manager closed")