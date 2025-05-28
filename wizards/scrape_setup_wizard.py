# wizards/scrape_setup_wizard.py

import yaml
import os
from datetime import datetime


def ask_question(prompt_text, default_value=None, required=True):
    """Helper to ask a question and get an answer."""
    full_prompt = f"{prompt_text}"
    if default_value is not None:
        full_prompt += f" (default: {default_value})"
    full_prompt += ": "

    while True:
        answer = input(full_prompt).strip()
        if answer:
            return answer
        elif default_value is not None:
            return default_value
        elif not required:
            return None
        else:
            print("This field is required. Please provide a value.")


def ask_for_list(prompt_text, item_name="item"):
    """Helper to ask for a list of items."""
    items = []
    print(f"\n--- {prompt_text} ---")
    while True:
        item = input(f"Enter a {item_name} (or press Enter to finish): ").strip()
        if not item:
            if not items:  # Require at least one if it's a list of seeds, for example
                print(f"Please add at least one {item_name}.")
                continue
            break
        items.append(item)
    return items


def main_wizard():
    print("🚀 Welcome to the Scrape Setup Wizard! 🚀")
    print("This wizard will help you create a configuration file for a new scraping domain.\n")

    domain_name_default = "my_new_scrape_domain"
    domain_name_user = ask_question(
        "Enter a general name for this scraping domain (e.g., 'Tennis Stats', 'Python Libraries')", domain_name_default)
    domain_slug = "".join(c if c.isalnum() else "_" for c in domain_name_user.lower())

    config_data = {
        'domain_info': {
            'name': domain_name_user,
            'wizard_generated_at': datetime.now().isoformat()
        },
        'sources': []
    }

    while True:
        add_another_source = ask_question("\nDo you want to add a new data source for this domain? (yes/no)",
                                          "yes").lower()
        if add_another_source not in ['yes', 'y']:
            break

        print("\n--- Configuring a New Data Source ---")

        source_name_default = f"source_{len(config_data['sources']) + 1}"
        source_name = ask_question("Unique name for this source (e.g., 'wikipedia_tennis_players', 'atp_results')",
                                   source_name_default)

        seed_urls = ask_for_list("Enter seed URLs for this source (where scraping will begin)", "Seed URL")

        print("\n--- CSS Selectors / XPath ---")
        print("For each item, provide the CSS selector or XPath to extract it.")
        print("If not applicable for this source, press Enter to skip.")

        selector_title = ask_question("Selector for 'Title'", required=False)
        selector_main_content = ask_question(
            "Selector for 'Main Content Body' (e.g., article text, product description)", required=False)
        selector_links_to_follow = ask_question("Selector for 'Links to Follow' (for crawling deeper)", required=False)
        # Add more common selectors if needed (date, author, etc.)
        custom_selectors = {}
        while True:
            add_custom_selector = ask_question("Add a custom selector for a specific field? (yes/no)", "no").lower()
            if add_custom_selector not in ['yes', 'y']:
                break
            field_key = ask_question("Enter a key for this custom field (e.g., 'player_ranking', 'match_score')")
            field_selector = ask_question(f"Enter selector for '{field_key}'")
            custom_selectors[field_key] = field_selector

        print("\n--- Crawl Settings ---")
        crawl_depth = ask_question("Crawl depth (e.g., 0 for just seed URLs, 1 for one level deeper)", "1")
        try:
            crawl_depth = int(crawl_depth)
        except ValueError:
            print("Invalid crawl depth, defaulting to 1.")
            crawl_depth = 1

        rate_limit = ask_question("Rate limit (e.g., '1/s' for 1 req/sec, '5/m' for 5 req/min)", "1/s")

        print("\n--- Export Settings ---")
        export_format = ask_question("Export format (e.g., jsonl, markdown, yaml)", "jsonl")
        output_path_default = f"./data/{domain_slug}/{source_name}.{export_format}"
        export_output_path = ask_question("Output path for exported data", output_path_default)

        source_entry = {
            'name': source_name,
            'seeds': seed_urls,
            'selectors': {},  # Initialize selectors dict
            'crawl': {
                'depth': crawl_depth,
                'rate_limit': rate_limit,
                # 'user_agent': "MyCustomScraper/1.0", # Could be an advanced option
                # 'headers': {}, # Advanced option
                # 'robots_txt': "honor" # Advanced option
            },
            'export': {
                'format': export_format,
                'output_path': export_output_path
            }
        }

        if selector_title: source_entry['selectors']['title'] = selector_title
        if selector_main_content: source_entry['selectors']['main_content'] = selector_main_content
        if selector_links_to_follow: source_entry['selectors']['links_to_follow'] = selector_links_to_follow
        if custom_selectors: source_entry['selectors']['custom_fields'] = custom_selectors

        config_data['sources'].append(source_entry)
        print(f"Source '{source_name}' added successfully!")

    if not config_data['sources']:
        print("\nNo data sources were configured. Exiting wizard.")
        return

    print("\n--- Generating Configuration File ---")
    output_dir = "configs"  # Suggest saving to a 'configs' directory
    os.makedirs(output_dir, exist_ok=True)

    config_filename_default = f"{domain_slug}_config.yaml"
    config_filename_user = ask_question(f"Enter filename for the generated config (e.g., {config_filename_default})",
                                        config_filename_default)

    full_output_path = os.path.join(output_dir, config_filename_user)

    try:
        with open(full_output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, sort_keys=False, allow_unicode=True, indent=2, default_flow_style=False)
        print(f"\n✅ Success! Configuration file generated at: {full_output_path}")
        print("You can now use this file with your scraper's ConfigManager.")
    except Exception as e:
        print(f"\n❌ Error! Could not write configuration file: {e}")


if __name__ == "__main__":
    main_wizard()