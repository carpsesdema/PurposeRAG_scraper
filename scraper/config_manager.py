# scraper/config_manager.py
from urllib.parse import urlparse

import yaml
import os
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, HttpUrl, Field, field_validator
import logging
import json  # <<< ADDED: Import the json module

import config


# --- Pydantic Models for Configuration Validation ---

class CustomFieldConfig(BaseModel):
    name: str = Field(...,
                      description="The meaningful name for this custom extracted field (e.g., 'article_author', 'match_score', 'player_ranking_entry').")
    selector: str = Field(..., description="CSS selector or XPath expression to locate the data.")
    extract_type: str = Field(default="text",
                              description="Type of data to extract: 'text', 'attribute', 'html', 'structured_list'.")
    attribute_name: Optional[str] = Field(default=None,
                                          description="If extract_type is 'attribute', specify the attribute (e.g., 'href', 'content', 'datetime').")
    is_list: bool = Field(default=False,
                          description="Set to true if the selector is expected to return multiple elements, yielding a list of values (especially for 'text', 'attribute', 'html'). For 'structured_list', this is implied true.")
    sub_selectors: Optional[List['CustomFieldConfig']] = Field(default=None,
                                                               description="For 'structured_list', defines fields to extract from each item found by the main selector.")

    @field_validator('extract_type')
    @classmethod
    def validate_extract_type(cls, value: str) -> str:
        allowed_types = ['text', 'attribute', 'html', 'structured_list']
        if value not in allowed_types:
            raise ValueError(f"extract_type must be one of {allowed_types}")
        return value

    @field_validator('attribute_name')
    @classmethod
    def validate_attribute_name(cls, value: Optional[str], values: Any) -> Optional[str]:
        data = values if isinstance(values, dict) else values.data
        if data.get('extract_type') == 'attribute' and not value:
            raise ValueError("attribute_name is required when extract_type is 'attribute'")
        return value

    @field_validator('sub_selectors')
    @classmethod
    def validate_sub_selectors(cls, value: Optional[List['CustomFieldConfig']], values: Any) -> Optional[
        List['CustomFieldConfig']]:
        data = values if isinstance(values, dict) else values.data
        if data.get('extract_type') == 'structured_list' and not value:
            raise ValueError("sub_selectors are required when extract_type is 'structured_list'")
        if data.get('extract_type') != 'structured_list' and value:
            raise ValueError("sub_selectors are only_applicable when extract_type is 'structured_list'")
        return value


class SelectorConfig(BaseModel):
    title: Optional[str] = None
    main_content: Optional[str] = Field(default=None,
                                        description="Selector for the main textual content area if Trafilatura isn't sufficient or for specific sections.")
    links_to_follow: Optional[str] = Field(default=None,
                                           description="Selector for links to be added to the crawl queue.")
    custom_fields: List[CustomFieldConfig] = Field(default_factory=list,
                                                   description="List of specific data points to extract with their selectors.")


class CrawlConfig(BaseModel):
    depth: int = 0
    delay_seconds: float = Field(default=1.0, description="Seconds to wait between requests to this source.")
    user_agent: Optional[str] = None
    respect_robots_txt: bool = True


class ExportConfig(BaseModel):
    format: str = "jsonl"
    output_path: str

    @field_validator('format')
    @classmethod
    def validate_export_format(cls, value: str) -> str:
        if value.lower() not in config.DEFAULT_EXPORT_FORMATS_SUPPORTED:
            raise ValueError(
                f"Unsupported export format: {value}. Supported: {config.DEFAULT_EXPORT_FORMATS_SUPPORTED}")
        return value.lower()


class SourceConfig(BaseModel):
    name: str = Field(..., description="Unique name for this data source (e.g., 'sofascore_match_reports')")
    seeds: List[HttpUrl] = Field(..., min_length=1, description="List of starting URLs for this source.")
    source_type: Optional[str] = Field(default=None,
                                       description="User-defined type hint, e.g., 'tennis_news_article', 'player_bio', 'tournament_draw'. Helps in analysis.")
    selectors: SelectorConfig = Field(default_factory=SelectorConfig)
    crawl_config: CrawlConfig = Field(default_factory=CrawlConfig, alias="crawl")
    export_config: ExportConfig = Field(..., alias="export")


class DomainScrapeConfig(BaseModel):
    domain_info: Dict[str, Any] = Field(default_factory=dict,
                                        description="General information about the domain/project.")
    sources: List[SourceConfig] = Field(..., min_length=1)
    global_user_agent: Optional[str] = None


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None, logger_instance=None):
        self.logger = logger_instance if logger_instance else logging.getLogger("ConfigManager_Fallback")
        self.config_path = config_path
        self.config: Optional[DomainScrapeConfig] = None

        if self.config_path:
            self.logger.info(f"ConfigManager initialized with path: {self.config_path}")
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

            # <<< MODIFIED DEBUG LINE >>>
            try:
                # Pydantic V2 style with indent if supported
                debug_json_str = self.config.model_dump_json(indent=2)
                self.logger.debug(f"Full loaded config (V2 style):\n{debug_json_str}")
            except TypeError:
                # Fallback for Pydantic V1 or V2 without indent in model_dump_json
                config_dict = self.config.model_dump(mode='json')  # Get dict suitable for JSON
                debug_json_str = json.dumps(config_dict, indent=2)  # Use json module for indentation
                self.logger.debug(f"Full loaded config (V1/fallback style):\n{debug_json_str}")
            # <<< END OF MODIFIED DEBUG LINE >>>

            return True
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e_yaml:
            self.logger.error(f"YAML parsing error in {self.config_path}: {e_yaml}")
        except Exception as e_val:  # Catches Pydantic validation errors and others
            # The logger.error in run_professional_pipeline will also catch this if load_config returns False
            # So, logging it here ensures it's captured specifically during the load attempt.
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

    def get_crawl_config_for_source(self, source_name: str) -> CrawlConfig:
        source = self.get_source_by_name(source_name)
        crawl_conf = source.crawl_config if source else CrawlConfig()
        if self.config and self.config.global_user_agent and not crawl_conf.user_agent:
            crawl_conf.user_agent = self.config.global_user_agent
        return crawl_conf

    def get_selectors_for_source(self, source_name: str) -> Optional[SelectorConfig]:
        source = self.get_source_by_name(source_name)
        return source.selectors if source else None

    def get_export_config_for_source(self, source_name: str) -> Optional[ExportConfig]:
        source = self.get_source_by_name(source_name)
        return source.export_config if source else None

    def get_site_config_for_url(self, url: str) -> Optional[SourceConfig]:
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


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as app_config

    logging.basicConfig(level=logging.DEBUG)
    test_logger = logging.getLogger(__name__)

    dummy_yaml_content_structured = f"""
domain_info:
  name: "Tennis Player Rankings (Test with Alias)"
global_user_agent: "RankingScraperBot/1.0"
sources:
  - name: "atp_rankings_detailed"
    seeds:
      - "https://www.atptour.com/en/rankings/singles"
    source_type: "player_rankings_table"
    selectors:
      custom_fields:
        - name: "rankings_data"
          selector: "table.desktop-table > tbody > tr"
          extract_type: "structured_list"
          is_list: true
          sub_selectors:
            - name: "rank"
              selector: "td.rank-cell"
              extract_type: "text"
            - name: "player_name"
              selector: "td.player-cell a"
              extract_type: "text"
    crawl: 
      depth: 0
      delay_seconds: 1.5
    export: 
      format: "jsonl"
      output_path: "./data_exports/tennis_domain/atp_rankings_data_alias.jsonl"
"""
    dummy_config_path_structured_alias = "dummy_tennis_rankings_config_alias.yaml"
    with open(dummy_config_path_structured_alias, 'w', encoding='utf-8') as f:
        f.write(dummy_yaml_content_structured)

    cfg_manager_structured_alias = ConfigManager(config_path=dummy_config_path_structured_alias,
                                                 logger_instance=test_logger)

    if cfg_manager_structured_alias.config:
        test_logger.info(
            f"Successfully loaded structured with alias: {cfg_manager_structured_alias.config.domain_info.get('name')}")
        rankings_source_alias = cfg_manager_structured_alias.get_source_by_name("atp_rankings_detailed")
        if rankings_source_alias:
            test_logger.info(f"Export format from alias config: {rankings_source_alias.export_config.format}")
            if rankings_source_alias.selectors and rankings_source_alias.selectors.custom_fields:
                cf_alias = rankings_source_alias.selectors.custom_fields[0]
                test_logger.info(f"Source custom field name (alias): {cf_alias.name}, type: {cf_alias.extract_type}")
                if cf_alias.sub_selectors:
                    test_logger.info(f"  Sub-selectors count (alias): {len(cf_alias.sub_selectors)}")
    else:
        test_logger.error("Failed to load structured dummy config with alias.")

    if os.path.exists(dummy_config_path_structured_alias):
        os.remove(dummy_config_path_structured_alias)