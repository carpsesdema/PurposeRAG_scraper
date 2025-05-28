import time
import requests
from typing import Callable, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from trafilatura import fetch_url

from .rag_models import FetchedItem

from config import USER_AGENT, DEFAULT_REQUEST_TIMEOUT


class RequestsDriver:
    def __init__(self, logger):
        self.logger = logger

    def fetch(self, url: str, source_type: str, query_used: str) -> Optional[FetchedItem]:
        self.logger.info(f"Fetching URL: {url} with RequestsDriver for source: {source_type}")
        try:
            content = fetch_url(url, timeout=DEFAULT_REQUEST_TIMEOUT, logger=self.logger)
            return FetchedItem.create(source_url=url, content=content, source_type=source_type, query_used=query_used)
        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP error fetching {url}: {http_err}")
        except requests.exceptions.RequestException as req_err:
            self.logger.error(f"Request error fetching {url}: {req_err}")
        except Exception as e:
            self.logger.error(f"Generic error fetching {url}: {e}", exc_info=True)
        return None


class FetcherPool:
    def __init__(self, num_workers: int, logger):
        self.num_workers = num_workers
        self.logger = logger
        self.driver = RequestsDriver(logger)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.futures = []

    def submit_task(self, url: str, source_type: str, query_used: str):
        """Submits a URL to be fetched."""
        self.logger.info(f"FetcherPool: Submitting task for URL: {url} (Source: {source_type})")
        future = self.executor.submit(self.driver.fetch, url, source_type, query_used)
        self.futures.append(future)

    def get_results(self) -> list[FetchedItem]:
        """Retrieves all fetched items, waiting for completion."""
        results = []
        for future in as_completed(self.futures):
            try:
                item = future.result()
                if item:
                    results.append(item)
            except Exception as e:
                self.logger.error(f"Exception in a fetcher task: {e}", exc_info=True)

        self.futures = []
        self.logger.info(f"FetcherPool: Retrieved {len(results)} items.")
        return results

    def shutdown(self):
        self.logger.info("Shutting down FetcherPool executor.")
        self.executor.shutdown(wait=True)