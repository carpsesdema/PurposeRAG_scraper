# scraper/config_manager.py
from urllib.parse import urlparse

import yaml
import os
from typing import Dict, List, Any, Optional, Union  # Added Union
from pydantic import BaseModel, HttpUrl, Field, validator, field_validator  # Added field_validator for Pydantic v2+
import logging  # Ensure logging is imported

import config


# --- Pydantic Models for Configuration Validation ---

class CustomFieldConfig(BaseModel):
    name: str = Field(...,
                      description="The meaningful name for this custom extracted field (e.g., 'article_author', 'match_score').")
    selector: str = Field(..., description="CSS selector or XPath expression to locate the data.")
    extract_type: str = Field(default="text", description="Type of data to extract: 'text', 'attribute', 'html'.")
    attribute_name: Optional[str] = Field(default=None,
                                          description="If extract_type is 'attribute', specify the attribute (e.g., 'href', 'content', 'datetime').")
    is_list: bool = Field(default=False,
                          description="Set to true if the selector is expected to return multiple elements, yielding a list of values.")

    # Future enhancements:
    # data_type: Optional[str] = Field(default="string", description="Expected data type: 'string', 'integer', 'float', 'datetime'. For validation/conversion.")
    # datetime_format: Optional[str] = Field(default=None, description="If data_type is 'datetime', specify the format string (e.g., '%Y-%m-%d').")
    # required: bool = Field(default=False)
    # default_value: Optional[Any] = None

    @field_validator('extract_type')
    @classmethod
    def validate_extract_type(cls, value: str) -> str:
        allowed_types = ['text', 'attribute', 'html']  # 'html' to get inner/outer HTML of selected element
        if value not in allowed_types:
            raise ValueError(f"extract_type must be one of {allowed_types}")
        return value

    @field_validator('attribute_name')
    @classmethod
    def validate_attribute_name(cls, value: Optional[str], values: Any) -> Optional[str]:
        # Pydantic v2: values is a FieldValidationInfo object, access data via values.data
        # For simplicity, assuming values is a dict if not using FieldValidationInfo structure explicitly
        data = values if isinstance(values, dict) else values.data
        if data.get('extract_type') == 'attribute' and not value:
            raise ValueError("attribute_name is required when extract_type is 'attribute'")
        return value


class SelectorConfig(BaseModel):
    # Predefined common selectors (can still be used for general purpose extraction)
    title: Optional[str] = None
    main_content: Optional[str] = Field(default=None,
                                        description="Selector for the main textual content area if Trafilatura isn't sufficient or for specific sections.")
    links_to_follow: Optional[str] = Field(default=None,
                                           description="Selector for links to be added to the crawl queue.")
    # publication_date: Optional[str] = None # Example of moving to CustomFieldConfig
    # author: Optional[str] = None         # Example of moving to CustomFieldConfig

    # New: List of custom field configurations
    custom_fields: List[CustomFieldConfig] = Field(default_factory=list,
                                                   description="List of specific data points to extract with their selectors.")


class CrawlConfig(BaseModel):
    depth: int = 0
    # rate_limit: str = "1/s" # Consider using a float for seconds per request
    delay_seconds: float = Field(default=1.0, description="Seconds to wait between requests to this source.")
    user_agent: Optional[str] = None
    respect_robots_txt: bool = True


class ExportConfig(BaseModel):
    format: str = "jsonl"  # For RAG chunks
    output_path: str  # For RAG chunks, e.g., "./data/domain_slug/source_name_rag.jsonl"

    @field_validator('format')
    @classmethod
    def validate_export_format(cls, value: str) -> str:
        if value.lower() not in config.DEFAULT_EXPORT_FORMATS_SUPPORTED:  # Assuming config.py has this list
            raise ValueError(
                f"Unsupported export format: {value}. Supported: {config.DEFAULT_EXPORT_FORMATS_SUPPORTED}")
        return value.lower()

    # Removed output_path validator as it's context-dependent.
    # Path creation logic is better handled in Exporter or when generating configs.


class SourceConfig(BaseModel):
    name: str = Field(..., description="Unique name for this data source (e.g., 'sofascore_match_reports')")
    seeds: List[HttpUrl] = Field(..., min_length=1, description="List of starting URLs for this source.")
    source_type: Optional[str] = Field(default=None,
                                       description="User-defined type hint, e.g., 'tennis_news_article', 'player_bio', 'tournament_draw'. Helps in analysis.")
    selectors: SelectorConfig = Field(default_factory=SelectorConfig)  # Updated SelectorConfig
    crawl_config: CrawlConfig = Field(default_factory=CrawlConfig, alias="crawl")
    export_config: ExportConfig  # Each source should define its RAG export

    # Could add source-level tags or default metadata here
    # default_tags: List[str] = Field(default_factory=list)


class DomainScrapeConfig(BaseModel):
    domain_info: Dict[str, Any] = Field(default_factory=dict,
                                        description="General information about the domain/project.")
    sources: List[SourceConfig] = Field(..., min_length=1)
    global_user_agent: Optional[str] = None
    # Removed global_rate_limit, use CrawlConfig.delay_seconds per source

    # Optional: Global export settings if not defined per source (less likely now)
    # global_export_config: Optional[ExportConfig] = None


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None, logger_instance=None):  # Use logger_instance
        self.logger = logger_instance if logger_instance else logging.getLogger("ConfigManager_Fallback")
        self.config_path = config_path
        self.config: Optional[DomainScrapeConfig] = None

        if self.config_path:
            self.load_config(self.config_path)
        else:
            self.logger.info("No config path for ConfigManager. Using defaults or programmatic config.")

    def load_config(self, config_path: str) -> bool:
        self.config_path = config_path
        self.logger.info(f"Loading configuration from: {self.config_path}")
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            self.config = DomainScrapeConfig(**raw_config)
            self.logger.info(f"Config loaded: {self.config.domain_info.get('name', 'Unknown Domain')}")
            self.logger.debug(f"Full loaded config: {self.config.model_dump(mode='json', indent=2)}")
            return True
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e_yaml:
            self.logger.error(f"YAML parsing error in {self.config_path}: {e_yaml}")
        except Exception as e_val:  # Catches Pydantic validation errors too
            self.logger.error(f"Validation/load error for config {self.config_path}: {e_val}", exc_info=True)
        self.config = None
        return False

    def get_sources(self) -> List[SourceConfig]:
        return self.config.sources if self.config else []

    def get_source_by_name(self, name: str) -> Optional[SourceConfig]:
        if not self.config: return None
        for source in self.config.sources:
            if source.name == name: return source
        self.logger.warning(f"Source '{name}' not found in configuration.")
        return None

    def get_crawl_config_for_source(self, source_name: str) -> CrawlConfig:  # Return default if not found
        source = self.get_source_by_name(source_name)
        crawl_conf = source.crawl_config if source else CrawlConfig()
        if self.config and self.config.global_user_agent and not crawl_conf.user_agent:
            crawl_conf.user_agent = self.config.global_user_agent
        return crawl_conf

    def get_selectors_for_source(self, source_name: str) -> Optional[SelectorConfig]:
        source = self.get_source_by_name(source_name)
        return source.selectors if source else None  # Return None if source or selectors not found

    def get_export_config_for_source(self, source_name: str) -> Optional[ExportConfig]:
        source = self.get_source_by_name(source_name)
        return source.export_config if source else None

    # New helper for site-specific config to pass to ContentRouter
    def get_site_config_for_url(self, url: str) -> Optional[SourceConfig]:
        """
        Finds the SourceConfig that is most relevant for a given URL.
        This could be based on matching seed URL domains or more complex rules.
        For now, a simple approach: if a SourceConfig's seed URL domain matches the given URL's domain.
        """
        if not self.config:
            return None

        try:
            parsed_target_url = urlparse(url)
            target_domain = parsed_target_url.netloc
        except Exception:
            self.logger.debug(f"Could not parse target URL for site config lookup: {url}")
            return None

        for source_config in self.config.sources:
            for seed_httpurl in source_config.seeds:
                seed_url_str = str(seed_httpurl)
                try:
                    parsed_seed_url = urlparse(seed_url_str)
                    if parsed_seed_url.netloc == target_domain:
                        self.logger.debug(
                            f"Found matching SourceConfig '{source_config.name}' for URL {url} based on domain.")
                        return source_config
                except Exception:
                    self.logger.debug(f"Could not parse seed URL {seed_url_str} during site config lookup.")
                    continue

        self.logger.debug(f"No specific SourceConfig found for URL {url} domain '{target_domain}'.")
        return None


# Example Usage (for testing this module)
if __name__ == "__main__":
    # Assuming config.py exists in the parent directory or PYTHONPATH is set
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as app_config  # For DEFAULT_EXPORT_FORMATS_SUPPORTED

    logging.basicConfig(level=logging.DEBUG)
    test_logger = logging.getLogger(__name__)

    dummy_yaml_content = f"""
domain_info:
  name: "Tennis Data Aggregation (Test)"
  version: "1.0"
global_user_agent: "TennisScraperBot/1.0 (+http://mycoolproject.com/botinfo)"
sources:
  - name: "tennis_news_site_x"
    seeds:
      - "https://www.example-tennisnews.com/articles"
    source_type: "tennis_news_article"
    selectors:
      title: "h1.article-title" # Standard title selector
      main_content: "div.article-body" # If Trafilatura needs help or for specific part
      custom_fields:
        - name: "publication_date"
          selector: "meta[property='article:published_time']"
          extract_type: "attribute"
          attribute_name: "content"
        - name: "author_name"
          selector: "span.author-name a" # Assuming author name is in an <a> tag
          extract_type: "text"
        - name: "category_tags"
          selector: "div.article-tags a.tag"
          extract_type: "text"
          is_list: true
    crawl:
      depth: 1
      delay_seconds: 2.5
    export: # RAG export for this source
      format: "jsonl"
      output_path: "./data_exports/tennis_domain/news_site_x_rag.jsonl"

  - name: "player_bio_site_y"
    seeds:
      - "https://www.example-playerbios.com/players"
    source_type: "player_biography"
    selectors:
      custom_fields:
        - name: "player_full_name"
          selector: "h1.player-profile-name"
          extract_type: "text"
        - name: "nationality"
          selector: "span.player-nationality-flag" # Assuming flag has text or nearby text
          extract_type: "text" # Or attribute if it's in, say, a title attribute
        - name: "career_high_ranking"
          selector: "div.stats-overview li#ranking-high span.value"
          extract_type: "text" # Could add data_type: "integer" later
    crawl:
      depth: 0 # Just seed pages
      delay_seconds: 3
    export: # RAG export for this source
      format: "markdown"
      output_path: "./data_exports/tennis_domain/player_bios_rag.md"
"""
    dummy_config_path = "dummy_tennis_config.yaml"
    with open(dummy_config_path, 'w', encoding='utf-8') as f:
        f.write(dummy_yaml_content)

    cfg_manager = ConfigManager(config_path=dummy_config_path, logger_instance=test_logger)

    if cfg_manager.config:
        test_logger.info(f"Successfully loaded: {cfg_manager.config.domain_info.get('name')}")
        news_source = cfg_manager.get_source_by_name("tennis_news_site_x")
        if news_source and news_source.selectors:
            test_logger.info(f"News source title selector: {news_source.selectors.title}")
            test_logger.info(f"News source custom fields count: {len(news_source.selectors.custom_fields)}")
            if news_source.selectors.custom_fields:
                test_logger.info(f"  First custom field name: {news_source.selectors.custom_fields[0].name}")
                test_logger.info(f"    Selector: {news_source.selectors.custom_fields[0].selector}")

        retrieved_config = cfg_manager.get_site_config_for_url(
            "https://www.example-tennisnews.com/articles/some-article-here")
        if retrieved_config:
            test_logger.info(f"Retrieved config for URL via domain match: {retrieved_config.name}")
        else:
            test_logger.warning("Could not retrieve config for URL via domain match.")

    else:
        test_logger.error("Failed to load dummy config for testing.")

    if os.path.exists(dummy_config_path):
        os.remove(dummy_config_path)