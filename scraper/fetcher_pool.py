# scraper/fetcher_pool.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from config import USER_AGENT, DEFAULT_REQUEST_TIMEOUT
from .rag_models import FetchedItem


# Removed trafilatura import here as we'll use requests directly for more control
# from trafilatura import fetch_url # We will replace this


# Assuming logger is passed in and configured externally
# from utils.logger import get_logger


class RequestsDriver:
    def __init__(self, logger):
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self, url: str, source_type: str, query_used: str, item_title: Optional[str] = None) -> Optional[
        FetchedItem]:
        self.logger.info(f"Fetching URL: {url} with RequestsDriver for source: {source_type}")
        try:
            response = self.session.get(url, timeout=DEFAULT_REQUEST_TIMEOUT, allow_redirects=True)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

            content_bytes = response.content
            content_type_detected = response.headers.get('Content-Type', '').lower()
            encoding = response.encoding or response.apparent_encoding  # Guess encoding if not specified

            text_content: Optional[str] = None
            if content_bytes:
                try:
                    if encoding:
                        text_content = content_bytes.decode(encoding, errors='replace')
                    else:
                        # If no encoding, try UTF-8 as a common default, then fall back to replace
                        text_content = content_bytes.decode('utf-8', errors='replace')
                    self.logger.debug(
                        f"Successfully decoded content for {url} using encoding {encoding or 'utf-8 (guessed)'}")
                except Exception as e_decode:
                    self.logger.warning(
                        f"Could not decode content from {url} as text using encoding {encoding}: {e_decode}. Content stored as bytes.")
                    text_content = None  # Ensure text_content is None if decoding fails

            return FetchedItem(
                source_url=url,
                content=text_content,
                content_bytes=content_bytes,
                content_type_detected=content_type_detected,
                source_type=source_type,
                query_used=query_used,
                title=item_title,  # Pass along title if provided (e.g. from search results)
                encoding=encoding
            )

        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP error fetching {url}: {http_err.response.status_code} {http_err}")
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout error fetching {url} after {DEFAULT_REQUEST_TIMEOUT}s")
        except requests.exceptions.RequestException as req_err:
            self.logger.error(f"Request error fetching {url}: {req_err}")
        except Exception as e:
            self.logger.error(f"Generic error fetching {url}: {e}", exc_info=True)
        return None


class FetcherPool:
    def __init__(self, num_workers: int, logger):
        self.num_workers = num_workers
        self.logger = logger
        self.driver = RequestsDriver(logger)  # Initialize with a logger
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.futures = []

    def submit_task(self, url: str, source_type: str, query_used: str, item_title: Optional[str] = None):
        """Submits a URL to be fetched. Item_title can be passed from search results."""
        self.logger.info(f"FetcherPool: Submitting task for URL: {url} (Source: {source_type}, Title: {item_title})")
        # Pass item_title to the driver's fetch method
        future = self.executor.submit(self.driver.fetch, url, source_type, query_used, item_title)
        self.futures.append(future)

    def get_results(self) -> list[FetchedItem]:
        """Retrieves all fetched items, waiting for completion."""
        results = []
        # Using as_completed to process results as they become available
        for future in as_completed(self.futures):
            try:
                item = future.result()  # This can re-raise exceptions from the worker
                if item:
                    results.append(item)
            except Exception as e:
                # The error should have been logged within self.driver.fetch or by the future itself
                # but we can log that a task resulted in an error here too.
                self.logger.error(f"A fetcher task failed: {e}", exc_info=False)  # exc_info=False as driver logs it

        self.futures = []  # Clear futures list for next batch
        self.logger.info(f"FetcherPool: Retrieved {len(results)} items from this batch.")
        return results

    def shutdown(self):
        self.logger.info("Shutting down FetcherPool executor.")
        self.executor.shutdown(wait=True)  # Wait for all tasks to complete
