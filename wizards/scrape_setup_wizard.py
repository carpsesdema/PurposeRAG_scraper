# wizards/scrape_setup_wizard.py

import os
from datetime import datetime
import yaml


# Ensure config_manager models are importable for type hints if needed,
# but wizard primarily generates dicts for YAML.
# from scraper.config_manager import CustomFieldConfig # Not strictly needed for generation

def ask_question(prompt_text, default_value=None, required=True, to_lower=False):
    """Helper to ask a question and get an answer."""
    full_prompt = f"{prompt_text}"
    if default_value is not None:
        full_prompt += f" (default: {default_value})"
    full_prompt += ": "

    while True:
        answer = input(full_prompt).strip()
        if to_lower:
            answer = answer.lower()
        if answer:
            return answer
        elif default_value is not None:
            return default_value
        elif not required:
            return None
        else:
            print("This field is required. Please provide a value.")


def ask_for_list(prompt_text, item_name="item", required=True):
    """Helper to ask for a list of items."""
    items = []
    print(f"\n--- {prompt_text} ---")
    while True:
        item = input(f"Enter a {item_name} (or press Enter to finish): ").strip()
        if not item:
            if required and not items:
                print(f"Please add at least one {item_name}.")
                continue
            break
        items.append(item)
    return items


def define_simple_custom_field():
    """Defines a single simple custom field (not part of a structured list)."""
    print("\n--- Defining a Simple Custom Field ---")
    field_name = ask_question(
        "Enter a name for this custom field (e.g., 'article_publication_date', 'overall_page_rating')")
    field_selector = ask_question(f"Enter CSS selector for '{field_name}'")

    extract_type_options = ["text", "attribute", "html"]
    extract_type_prompt = f"What type of data to extract for '{field_name}'? ({'/'.join(extract_type_options)})"
    field_extract_type = ask_question(extract_type_prompt, default_value="text", to_lower=True)
    while field_extract_type not in extract_type_options:
        print(f"Invalid extract type. Please choose from: {', '.join(extract_type_options)}")
        field_extract_type = ask_question(extract_type_prompt, default_value="text", to_lower=True)

    field_attribute_name = None
    if field_extract_type == "attribute":
        field_attribute_name = ask_question("Enter the attribute name (e.g., 'href', 'src', 'content', 'datetime')")

    field_is_list_str = ask_question(
        f"Should this field '{field_name}' be a list of values (e.g., multiple tags)? (yes/no)", "no", to_lower=True)
    field_is_list = field_is_list_str in ['yes', 'y']

    custom_field_entry = {
        'name': field_name,
        'selector': field_selector,
        'extract_type': field_extract_type
    }
    if field_attribute_name:
        custom_field_entry['attribute_name'] = field_attribute_name
    if field_is_list:  # Only add is_list if true, as default is false in Pydantic model
        custom_field_entry['is_list'] = field_is_list

    return custom_field_entry


def define_structured_list_field():
    """Defines a custom field that extracts a list of structured items (e.g., table rows)."""
    print("\n--- Defining a Structured List Field (e.g., for table rows) ---")
    list_name = ask_question(
        "Enter a name for this list of structured items (e.g., 'player_rankings_entries', 'product_details')")
    item_selector = ask_question(
        f"Enter the CSS selector that identifies EACH REPEATING ITEM in the '{list_name}' list (e.g., 'table > tbody > tr', 'div.product-card')")

    sub_selectors = []
    print(f"\n--- Now, define the sub-fields to extract from EACH item within '{list_name}' ---")
    while True:
        add_sub_field = ask_question("Add a sub-field to extract from each item? (yes/no)", "yes", to_lower=True)
        if add_sub_field not in ['yes', 'y']:
            break

        print("\n  --- Defining a Sub-Field ---")
        sub_name = ask_question("  Name for this sub-field (e.g., 'rank', 'player_name', 'product_price')")
        sub_selector = ask_question(
            f"  CSS selector for '{sub_name}', relative to an item selected by '{item_selector}' (e.g., 'td.rank-cell', 'span.price')")

        extract_type_options = ["text", "attribute", "html"]
        extract_type_prompt = f"  What type of data for sub-field '{sub_name}'? ({'/'.join(extract_type_options)})"
        sub_extract_type = ask_question(extract_type_prompt, "text", to_lower=True)
        while sub_extract_type not in extract_type_options:
            print(f"  Invalid extract type. Please choose from: {', '.join(extract_type_options)}")
            sub_extract_type = ask_question(extract_type_prompt, "text", to_lower=True)

        sub_attribute_name = None
        if sub_extract_type == "attribute":
            sub_attribute_name = ask_question("  Enter the attribute name (e.g., 'href', 'src', 'title')")

        sub_selector_entry = {
            'name': sub_name,
            'selector': sub_selector,
            'extract_type': sub_extract_type
        }
        if sub_attribute_name:
            sub_selector_entry['attribute_name'] = sub_attribute_name
        # 'is_list' is typically false for sub-selectors within a structured item,
        # as they usually target a single piece of data per item.

        sub_selectors.append(sub_selector_entry)

    if not sub_selectors:
        print(f"No sub-fields defined for the structured list '{list_name}'. This field will be skipped.")
        return None

    structured_list_entry = {
        'name': list_name,
        'selector': item_selector,
        'extract_type': 'structured_list',  # Hardcoded for this function
        'is_list': True,  # Implied true, also helps Pydantic if it relies on it
        'sub_selectors': sub_selectors
    }
    return structured_list_entry


def main_wizard():
    print("🚀 Welcome to the ModularRAGScraper Setup Wizard! 🚀")
    print("This wizard will help you create a YAML configuration file for scraping specific data sources.\n")

    domain_name_default = "my_scrape_project"
    domain_name_user = ask_question(
        "Enter a general name for this scraping project/domain (e.g., 'Tennis Stats', 'E-commerce Products')",
        domain_name_default)
    domain_slug = "".join(c if c.isalnum() else "_" for c in domain_name_user.lower().replace(" ", "_"))

    config_data = {
        'domain_info': {
            'name': domain_name_user,
            'description': ask_question("Brief description of this project/domain", required=False),
            'wizard_generated_at': datetime.now().isoformat()
        },
        'global_user_agent': ask_question("Global User-Agent string (leave blank for default)",
                                          f"{domain_slug}_bot/1.0", required=False),
        'sources': []
    }

    while True:
        add_another_source = ask_question("\nDo you want to add a data source configuration? (yes/no)", "yes",
                                          to_lower=True)
        if add_another_source not in ['yes', 'y']:
            break

        print("\n--- Configuring a New Data Source ---")
        source_name_default = f"{domain_slug}_source_{len(config_data['sources']) + 1}"
        source_name = ask_question("Unique name for this source (e.g., 'atp_rankings', 'product_listings')",
                                   source_name_default)
        seed_urls = ask_for_list("Enter seed URLs for this source (where scraping will begin)", "Seed URL",
                                 required=True)
        source_type = ask_question(
            "Descriptive type for this source (e.g., 'atp_rankings_table', 'news_articles_list')", source_name)

        # --- Selectors ---
        source_selectors_config = {}
        print("\n--- Defining Selectors for this Source ---")

        source_selectors_config['title'] = ask_question("CSS selector for the main page/item Title (optional)",
                                                        required=False)

        # Ask if Trafilatura's main content is enough, or if a specific selector is needed
        use_custom_main_content = ask_question(
            "Do you need a specific CSS selector for the 'Main Content Body' (if general text extraction isn't enough)? (yes/no)",
            "no", to_lower=True)
        if use_custom_main_content in ['yes', 'y']:
            source_selectors_config['main_content'] = ask_question("CSS selector for 'Main Content Body'",
                                                                   required=True)

        source_selectors_config['links_to_follow'] = ask_question(
            "CSS selector for 'Links to Follow' for crawling (optional, if crawl depth > 0)", required=False)

        all_custom_fields_for_source = []
        # Simple Custom Fields
        while True:
            add_simple_cf = ask_question(
                "Add a 'Simple Custom Field' (single value or simple list from the page)? (yes/no)", "no",
                to_lower=True)
            if add_simple_cf not in ['yes', 'y']:
                break
            simple_cf = define_simple_custom_field()
            if simple_cf:
                all_custom_fields_for_source.append(simple_cf)

        # Structured List Fields
        while True:
            add_structured_cf = ask_question(
                "Add a 'Structured List Field' (e.g., to extract table rows or product cards)? (yes/no)", "no",
                to_lower=True)
            if add_structured_cf not in ['yes', 'y']:
                break
            structured_cf = define_structured_list_field()
            if structured_cf:
                all_custom_fields_for_source.append(structured_cf)

        if all_custom_fields_for_source:
            source_selectors_config['custom_fields'] = all_custom_fields_for_source

        # Clean up empty selector fields
        final_selectors_config = {k: v for k, v in source_selectors_config.items() if v}

        # --- Crawl Settings ---
        print("\n--- Crawl Settings for this Source ---")
        crawl_depth_str = ask_question("Crawl depth (0 for just seed URLs, 1 for one level deeper, etc.)", "0")
        try:
            crawl_depth = int(crawl_depth_str)
        except ValueError:
            print("Invalid crawl depth, defaulting to 0.")
            crawl_depth = 0

        crawl_delay_str = ask_question("Delay between requests to this source (seconds, e.g., 1.0, 2.5)", "2.0")
        try:
            crawl_delay = float(crawl_delay_str)
        except ValueError:
            print("Invalid delay, defaulting to 2.0 seconds.")
            crawl_delay = 2.0

        respect_robots_str = ask_question("Respect robots.txt for this source? (yes/no)", "yes", to_lower=True)
        respect_robots = respect_robots_str in ['yes', 'y']

        crawl_config = {
            'depth': crawl_depth,
            'delay_seconds': crawl_delay,
            'respect_robots_txt': respect_robots
        }
        # User agent can be added here if needed per source, overriding global

        # --- Export Settings ---
        print("\n--- Export Settings for this Source ---")
        export_format_options = ["jsonl", "markdown"]  # Assuming these are supported by your Exporter
        export_format_prompt = f"Export format for RAG chunks from this source ({'/'.join(export_format_options)})"
        export_format = ask_question(export_format_prompt, "jsonl", to_lower=True)
        while export_format not in export_format_options:
            print(f"Invalid export format. Please choose from: {', '.join(export_format_options)}")
            export_format = ask_question(export_format_prompt, "jsonl", to_lower=True)

        output_path_default = f"./data_exports/{domain_slug}/{source_name}_rag.{export_format}"
        export_output_path = ask_question("Output path for exported RAG data", output_path_default)

        export_config = {
            'format': export_format,
            'output_path': export_output_path
        }

        source_entry = {
            'name': source_name,
            'seeds': seed_urls,
            'source_type': source_type,
            'selectors': final_selectors_config,  # Add the populated selectors
            'crawl': crawl_config,  # Renamed to 'crawl' to match alias
            'export': export_config  # Renamed to 'export' to match alias
        }
        config_data['sources'].append(source_entry)
        print(f"\nSource '{source_name}' added successfully!")

    if not config_data['sources']:
        print("\nNo data sources were configured. Exiting wizard.")
        return

    print("\n--- Generating Configuration File ---")
    output_dir = "configs"
    os.makedirs(output_dir, exist_ok=True)

    config_filename_default = f"{domain_slug}_config.yaml"
    config_filename_user = ask_question(f"Enter filename for the generated config (e.g., {config_filename_default})",
                                        config_filename_default)
    full_output_path = os.path.join(output_dir, config_filename_user)

    try:
        with open(full_output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, sort_keys=False, allow_unicode=True, indent=2, default_flow_style=False)
        print(f"\n✅ Success! Configuration file generated at: {full_output_path}")
        print("You can now use this file path in the GUI's 'Search Query / File Path' input field.")
    except Exception as e:
        print(f"\n❌ Error! Could not write configuration file: {e}")


if __name__ == "__main__":
    main_wizard()