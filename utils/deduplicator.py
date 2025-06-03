# utils/deduplicator.py
import hashlib
import logging
from typing import Set, Tuple, Optional, Dict, Any


class SmartDeduplicator:
    """
    Simple but effective deduplication system for scraped content.
    Uses MD5 hashing for fast duplicate detection.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.seen_hashes: Set[str] = set()
        self.logger = logger or logging.getLogger("SmartDeduplicator")
        self.stats = {
            'total_checked': 0,
            'duplicates_found': 0,
            'unique_content': 0
        }

    def add_snippet(self, text_content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add content to the deduplication set.
        Returns True if content was unique (added), False if duplicate.
        """
        if not text_content or not text_content.strip():
            return False

        content_hash = self._generate_hash(text_content)
        self.stats['total_checked'] += 1

        if content_hash in self.seen_hashes:
            self.stats['duplicates_found'] += 1
            self.logger.debug(f"Duplicate content detected (hash: {content_hash[:8]}...)")
            return False
        else:
            self.seen_hashes.add(content_hash)
            self.stats['unique_content'] += 1
            self.logger.debug(f"Unique content added (hash: {content_hash[:8]}...)")
            return True

    def is_duplicate(self, text_content: str) -> Tuple[bool, str]:
        """
        Check if content is a duplicate without adding it.
        Returns (is_duplicate, reason)
        """
        if not text_content or not text_content.strip():
            return True, "empty_content"

        content_hash = self._generate_hash(text_content)

        if content_hash in self.seen_hashes:
            return True, "exact_hash_match"
        else:
            return False, "unique_content"

    def _generate_hash(self, text_content: str) -> str:
        """Generate MD5 hash of normalized text content."""
        # Normalize text: lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(text_content.lower().split())
        return hashlib.md5(normalized.encode('utf-8', errors='replace')).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            **self.stats,
            'duplicate_rate': (self.stats['duplicates_found'] / max(self.stats['total_checked'], 1)) * 100,
            'unique_rate': (self.stats['unique_content'] / max(self.stats['total_checked'], 1)) * 100
        }

    def clear(self):
        """Clear all stored hashes and reset stats."""
        self.seen_hashes.clear()
        self.stats = {
            'total_checked': 0,
            'duplicates_found': 0,
            'unique_content': 0
        }
        self.logger.info("Deduplicator cleared")


# Backwards compatibility
def create_deduplicator(logger=None) -> SmartDeduplicator:
    """Factory function for creating deduplicator instances."""
    return SmartDeduplicator(logger=logger)