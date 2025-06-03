# Modular RAG Content Scraper (PurposeRAG_scraper)

The Modular RAG Content Scraper is a Python application designed for domain-agnostic information retrieval. It features a graphical user interface (GUI) built with PySide6, a configurable scraping pipeline, and tools for processing and exporting content suitable for Retrieval Augmented Generation (RAG) systems.

## Features

* **Graphical User Interface:** Easy-to-use interface for initiating scrapes and viewing progress.
* **Configurable Scraping:** Define scrape targets, depths, and selectors using YAML configuration files.
* **Content Processing:** Includes fetching, parsing (HTML, PDF, JSON, XML), normalization, deduplication, and content enrichment (NLP-based tagging, categorization, entity extraction if NLP libraries are available).
* **RAG-Optimized Output:** Chunks content and exports it in formats like JSONL or Markdown, ready for RAG pipelines.
* **Autonomous Search:** Can use DuckDuckGo (if available) to find relevant URLs based on a search query.
* **Local File Processing:** Can process local files by providing their path.
* **Extensible:** Designed with a modular structure for easier extension (e.g., adding new fetch drivers, parsers, or export formats).

## Project Structure Overview

    PurposeRAG_scraper/
    ├── configs/                  # YAML configuration files for scraping tasks
    │   └── tennis_config.yaml    # Example configuration
    ├── data_exports/             # Default directory for exported RAG data
    ├── gui/
    │   ├── main_window.py        # Main application window (GUI)
    │   └── styles.qss            # Stylesheet for the GUI
    ├── scraper/
    │   ├── chunker.py            # Content chunking logic
    │   ├── config_manager.py     # Handles loading and validation of YAML configs
    │   ├── content_router.py     # Routes fetched content to appropriate parsers
    │   ├── fetcher_pool.py       # Manages concurrent fetching of web content
    │   ├── parser.py             # HTML, PDF, and other content parsing utilities
    │   ├── rag_models.py         # Pydantic models for data structures
    │   └── searcher.py           # Core scraping pipeline logic
    ├── storage/
    │   └── saver.py              # Saves processed EnrichedItem data to disk
    ├── utils/
    │   ├── deduplicator.py       # Content deduplication utility
    │   └── logger.py             # Application logging setup
    ├── wizards/
    │   └── scrape_setup_wizard.py # CLI wizard to help create new scrape configurations
    ├── config.py                 # Central application configuration settings
    ├── main.py                   # Main entry point to run the GUI application
    ├── rag_scraper.log           # Log file
    └── requirements.txt          # Python package dependencies

## 1. Setting Up Your Python Environment

You'll need Python 3 installed on your system. This project uses a `requirements.txt` file to manage its dependencies.

### General Setup (Virtual Environment - Recommended)

It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects or your global Python installation.

1.  **Navigate to the project directory:**
    Open your terminal or command prompt and `cd` into the `PurposeRAG_scraper` directory.

    ```bash
    cd path/to/PurposeRAG_scraper
    ```

2.  **Create a virtual environment:**
    If you have `venv` (standard in Python 3.3+):

    ```bash
    python -m venv .venv
    ```
    (This creates a virtual environment in a folder named `.venv` within your project directory.)

3.  **Activate the virtual environment:**
    * On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    * On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    Your terminal prompt should now indicate that you are in the virtual environment (e.g., `(.venv) ...`).

4.  **Install dependencies:**
    With the virtual environment activated, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

    This will install PySide6, requests, BeautifulSoup4, lxml, trafilatura, PyYAML, spaCy, langdetect, duckduckgo-search, and pdfminer.six.

5.  **(Optional but Recommended) Download spaCy NLP model:**
    The application uses spaCy for NLP tasks like Named Entity Recognition (NER). You'll need to download a model. The default is `en_core_web_sm`.

    ```bash
    python -m spacy download en_core_web_sm
    ```
    If you change `SPACY_MODEL_NAME` in `config.py` to a different model, download that one instead.

Now, proceed with editor-specific setup.

### Using PyCharm

PyCharm has excellent built-in support for managing virtual environments.

1.  **Open the Project:**
    * Open PyCharm.
    * Select "Open" and navigate to the `PurposeRAG_scraper` directory.

2.  **Configure the Project Interpreter:**
    PyCharm usually detects `requirements.txt` and might prompt you to create a virtual environment or use an existing one.
    * If you've already created and activated the virtual environment as described in "General Setup", PyCharm should ideally detect it.
    * If not, or to verify:
        * Go to `File` -> `Settings` (or `PyCharm` -> `Preferences` on macOS).
        * Navigate to `Project: PurposeRAG_scraper` -> `Python Interpreter`.
        * Click the gear icon ⚙️ next to the Python Interpreter dropdown and select "Add...".
        * In the "Add Python Interpreter" dialog:
            * Choose "Virtualenv Environment".
            * Select "Existing environment".
            * For the "Interpreter" field, browse to the `python.exe` (Windows) or `python` (macOS/Linux) executable inside your `.venv/Scripts/` or `.venv/bin/` folder respectively.
            * Alternatively, if you haven't created one yet, you can choose "New environment" and PyCharm will create one for you (ensure the base interpreter is correct and it uses the project's `requirements.txt`).
        * Click "OK". PyCharm will now use this interpreter and its installed packages.

3.  **Install Dependencies (if not already done):**
    If you didn't run `pip install -r requirements.txt` in the terminal, PyCharm's terminal (View -> Tool Windows -> Terminal) will use the configured project interpreter (and its activated virtual environment). You can run the command there:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

4.  **Set up Run Configuration (Optional, for convenience):**
    * Click on "Add Configuration..." (usually near the top-right, next to the play button).
    * Click the "+" button and select "Python".
    * **Name:** `Run ModularRAGScraper GUI` (or any name you prefer).
    * **Script path:** Browse to and select `main.py` in your project.
    * **Python interpreter:** Ensure it's set to the project's virtual environment interpreter.
    * **Working directory:** Should be the root of your `PurposeRAG_scraper` project.
    * Click "OK". You can now run the application using this configuration.

### Using Visual Studio Code (VSCode)

VSCode is also great for Python development and works well with virtual environments.

1.  **Open the Project:**
    * Open VSCode.
    * Go to `File` -> `Open Folder...` and select the `PurposeRAG_scraper` directory.

2.  **Install Python Extension:**
    If you haven't already, install the official Python extension by Microsoft from the Extensions view (Ctrl+Shift+X).

3.  **Select Python Interpreter:**
    * Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS).
    * Type "Python: Select Interpreter".
    * VSCode should list available interpreters, including those in virtual environments. Select the Python interpreter located in your `.venv` folder (e.g., `.venv/bin/python` or `.venv\Scripts\python.exe`).
    * If it's not listed, you might need to enter the full path to the interpreter.
    * Once selected, VSCode will use this interpreter for this workspace. If you open a new terminal in VSCode (Terminal -> New Terminal), it should automatically activate the selected virtual environment.

4.  **Install Dependencies (if not already done):**
    Open a new terminal in VSCode (it should be using your virtual environment). Run:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

5.  **Set up Launch Configuration (Optional, for debugging/running):**
    * Go to the "Run and Debug" view (Ctrl+Shift+D or the icon on the sidebar).
    * If you see "create a launch.json file", click it.
    * Select "Python File" as the debug configuration. VSCode will create a `launch.json` file in a `.vscode` folder. It will typically be configured to run the currently open Python file.
    * To always run `main.py`:
        Modify `launch.json` to look something like this:
        ```json
        {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Run ModularRAGScraper GUI",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/main.py", 
                    "console": "integratedTerminal",
                    "justMyCode": true
                }
            ]
        }
        ```
    * You can now run or debug the application from the "Run and Debug" view by selecting this configuration and pressing F5.

## 2. Running the Application

Once your environment is set up and dependencies are installed:

1.  **Activate your virtual environment** (if it's not already activated by your IDE's terminal):
    * Windows: `.\.venv\Scripts\activate`
    * macOS/Linux: `source .venv/bin/activate`

2.  **Navigate to the project's root directory** (`PurposeRAG_scraper`).

3.  **Run the main application script:**
    ```bash
    python main.py
    ```
    This will launch the Modular RAG Content Scraper GUI.

### Using the Application

* **Mode:**
    * **Search Query / File Path:** Enter a search term (uses DuckDuckGo) or the full path to a local YAML configuration file (e.g., `configs/tennis_config.yaml`) or even a direct path to a single local file you want to process.
    * **Direct URL:** Enter a specific URL to fetch and process.
* **Content Type Hint:** Select the expected content type. 'Auto-Detect Content' is generally good, but you can provide a hint if you know the content type (e.g., 'PDF Document' for a direct PDF URL).
* **Fetch & Process:** Starts the scraping and processing pipeline.
* **Content Preview Tab:** Shows a preview of the fetched and primarily extracted text content.
* **Analysis & Metadata Tab:** Displays a summary of the enriched items, including metadata like categories, tags, and structured elements found.
* **Save Processed Content:** After fetching, this button allows you to save the structured output of each processed item (metadata, main text, and individual structured elements) to a directory you select.
* **RAG Export Info:** Provides information that RAG-optimized chunks (usually JSONL or Markdown) are automatically exported by the backend based on the configuration. Default export location is usually within `data_exports/`.

## 3. Creating Custom Scrape Configurations (YAML)

The application can be driven by YAML configuration files that define specific data sources, seed URLs, CSS/XPath selectors for custom data extraction, crawl depth, and export settings.

* An example configuration is provided in `configs/tennis_config.yaml`.
* You can create your own YAML files based on this structure.

### Using the Scrape Setup Wizard

To help create new YAML configuration files, a command-line wizard is available:

1.  **Activate your virtual environment.**
2.  **Navigate to the project root.**
3.  **Run the wizard:**
    ```bash
    python wizards/scrape_setup_wizard.py
    ```
    The wizard will guide you through defining sources, seed URLs, selectors, and other parameters, then generate a YAML file (usually in the `configs/` directory). You can then use the path to this generated YAML file in the GUI's "Search Query / File Path" input field.

## 4. Logging

* The application logs its activities to `rag_scraper.log` in the project root.
* Log levels for console and file output can be configured in `config.py`.

## 5. Configuration (`config.py`)

The `config.py` file contains various application-wide settings:
* Default logger names and log levels.
* GUI settings (window title, dimensions).
* Fetcher settings (timeout, user agent, concurrency).
* Enabled content types for processing.
* Chunking parameters for RAG output.
* And more.

Review and modify this file if you need to change default behaviors.

---

This README should give users a good starting point for setting up and running your project. Remember to keep it updated as your project evolves!