# scraper/config_manager.py

import yaml
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, HttpUrl, Field, validator


# --- Pydantic Models for Configuration Validation ---

class SelectorConfig(BaseModel):
    title: Optional[str] = None
    main_content: Optional[str] = None
    links_to_follow: Optional[str] = None
    publication_date: Optional[str] = None
    author: Optional[str] = None
    custom_fields: Optional[Dict[str, str]] = {}


class CrawlConfig(BaseModel):
    depth: int = 0
    rate_limit: str = "1/s"  # e.g., "1/s", "5/m"
    user_agent: Optional[str] = None
    respect_robots_txt: bool = True
    # headers: Optional[Dict[str, str]] = None # For future use


class ExportConfig(BaseModel):
    format: str = "jsonl"
    output_path: str  # e.g., "./data/domain_slug/source_name.jsonl"

    @validator('output_path')
    def output_path_must_be_relative_or_absolute(cls, v):
        # Basic check, can be expanded
        if not (v.startswith('./') or v.startswith('../') or os.path.isabs(v)):
            # If it's just a filename, prepend './data/' as a sensible default base
            # This aligns with how the wizard might generate paths.
            # For true flexibility, the user should provide full paths or clear relative paths.
            # For now, let's assume paths from wizard are either full or relative to project root.
            pass
        return v


class SourceConfig(BaseModel):
    name: str = Field(..., description="Unique name for this data source")
    seeds: List[HttpUrl] = Field(..., min_length=1, description="List of starting URLs for this source")
    source_type: Optional[str] = Field(None,
                                       description="Helps router decide parsing logic, e.g., 'html_article', 'pdf_report', 'forum'")
    selectors: Optional[SelectorConfig] = Field(default_factory=SelectorConfig)
    crawl_config: CrawlConfig = Field(default_factory=CrawlConfig, alias="crawl")  # alias for YAML friendliness
    export_config: ExportConfig = Field(..., alias="export")  # alias for YAML friendliness


class DomainScrapeConfig(BaseModel):
    domain_info: Dict[str, Any] = Field(default_factory=dict,
                                        description="General information about the domain being scraped")
    sources: List[SourceConfig] = Field(..., min_length=1)

    # Global settings that can override individual source settings or provide defaults
    global_user_agent: Optional[str] = None
    global_rate_limit: Optional[str] = None


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None, logger=None):
        self.logger = logger if logger else logging.getLogger("ConfigManager_Fallback")
        self.config_path = config_path
        self.config: Optional[DomainScrapeConfig] = None

        if self.config_path:
            self.load_config(self.config_path)
        else:
            self.logger.info(
                "No config path provided to ConfigManager. Scraper will need programmatic configuration or use global defaults.")

    def load_config(self, config_path: str) -> bool:
        self.config_path = config_path
        self.logger.info(f"Loading configuration from: {self.config_path}")
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            # Validate the loaded config using Pydantic model
            self.config = DomainScrapeConfig(**raw_config)
            self.logger.info(
                f"Configuration loaded successfully for domain: {self.config.domain_info.get('name', 'Unknown')}")
            self.logger.debug(f"Loaded config data: {self.config.model_dump_json(indent=2)}")
            return True
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found at: {self.config_path}")
            self.config = None
            return False
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration file {self.config_path}: {e}")
            self.config = None
            return False
        except Exception as e:  # Catches Pydantic validation errors too
            self.logger.error(f"Error loading or validating configuration from {self.config_path}: {e}", exc_info=True)
            self.config = None
            return False

    def get_sources(self) -> List[SourceConfig]:
        if not self.config:
            self.logger.warning("Attempted to get sources, but no configuration is loaded.")
            return []
        return self.config.sources

    def get_source_by_name(self, name: str) -> Optional[SourceConfig]:
        if not self.config:
            return None
        for source in self.config.sources:
            if source.name == name:
                return source
        self.logger.warning(f"Source with name '{name}' not found in configuration.")
        return None

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        if not self.config:
            return default
        # Example: access self.config.global_user_agent
        return getattr(self.config, key, default)

    # --- Helper methods to get specific parts of the config easily ---
    def get_seed_urls_for_source(self, source_name: str) -> List[HttpUrl]:
        source = self.get_source_by_name(source_name)
        return source.seeds if source else []

    def get_selectors_for_source(self, source_name: str) -> Optional[SelectorConfig]:
        source = self.get_source_by_name(source_name)
        return source.selectors if source else None

    def get_crawl_config_for_source(self, source_name: str) -> Optional[CrawlConfig]:
        source = self.get_source_by_name(source_name)
        crawl_conf = source.crawl_config if source else CrawlConfig()  # Default if source not found

        # Apply global overrides if they exist
        if self.config:
            if self.config.global_user_agent and crawl_conf.user_agent is None:
                crawl_conf.user_agent = self.config.global_user_agent
            if self.config.global_rate_limit and crawl_conf.rate_limit == CrawlConfig().rate_limit:  # if not overridden by source
                crawl_conf.rate_limit = self.config.global_rate_limit
        return crawl_conf

    def get_export_config_for_source(self, source_name: str) -> Optional[ExportConfig]:
        source = self.get_source_by_name(source_name)
        return source.export_config if source else None


# Example Usage (for testing this module)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Create a dummy config file for testing
    dummy_config_content = {
        "domain_info": {"name": "Test Tennis Scrape"},
        "global_user_agent": "GlobalScraperBot/1.1",
        "sources": [
            {
                "name": "wikipedia_tennis",
                "seeds": ["https://en.wikipedia.org/wiki/Tennis"],
                "source_type": "html_article",
                "selectors": {
                    "title": "h1#firstHeading",
                    "main_content": "div#mw-content-text"
                },
                "crawl": {  # Uses alias "crawl"
                    "depth": 1,
                    "rate_limit": "2/s"
                },
                "export": {  # Uses alias "export"
                    "format": "markdown",
                    "output_path": "./data/test_tennis/wiki_tennis.md"
                }
            }
        ]
    }
    dummy_config_path = "dummy_test_config.yaml"
    with open(dummy_config_path, 'w') as f:
        yaml.dump(dummy_config_content, f)

    cfg_manager = ConfigManager(config_path=dummy_config_path, logger=logger)

    if cfg_manager.config:
        logger.info(f"Domain: {cfg_manager.config.domain_info.get('name')}")
        sources = cfg_manager.get_sources()
        logger.info(f"Number of sources: {len(sources)}")
        for src in sources:
            logger.info(f"Source Name: {src.name}")
            logger.info(f"  Seeds: {src.seeds}")
            logger.info(f"  Selectors-Title: {src.selectors.title if src.selectors else 'N/A'}")
            logger.info(f"  Crawl Depth: {src.crawl_config.depth}")
            logger.info(
                f"  Crawl User Agent: {src.crawl_config.user_agent or cfg_manager.get_global_setting('global_user_agent')}")  # Shows applying global
            logger.info(f"  Export Path: {src.export_config.output_path}")
    else:
        logger.error("Failed to load dummy config for testing.")

    os.remove(dummy_config_path)  # Clean up