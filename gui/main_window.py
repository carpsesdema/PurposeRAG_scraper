# gui/main_window.py
import sys
import logging
from typing import List

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QPlainTextEdit, QFileDialog,
    QLabel, QComboBox, QProgressBar, QCheckBox,  # QCheckBox not used, can remove
    QGroupBox, QTextEdit, QTabWidget, QSplitter
)

from scraper.searcher import search_and_fetch
from scraper.rag_models import EnrichedItem  # <-- Import EnrichedItem
from storage.saver import save_enriched_items_to_disk
# <-- Updated import
from utils.logger import setup_logger
import config
import json  # For pretty printing metadata in GUI


class FetchWorker(QThread):
    progress = Signal(int, str)
    # MODIFIED: Emits List[EnrichedItem] and a status message
    finished = Signal(list, str)  # list will be List[EnrichedItem]
    error = Signal(str)

    def __init__(self, query, mode, content_type_for_gui, logger_instance):
        super().__init__()
        self.query = query
        self.mode = mode  # 'Search' or 'URL'
        self.content_type_for_gui = content_type_for_gui  # Hint like 'html', 'pdf', 'auto'
        self.logger = logger_instance
        # self.total_sources = config.SEARCH_SOURCES_COUNT # This seems like a legacy var

    def run(self):
        try:
            self.progress.emit(10,
                               f"Starting search for: {self.query} (Mode: {self.mode}, Type Hint: {self.content_type_for_gui})...")

            def backend_progress_callback(message, percentage_step):
                # Scale backend progress (0-100) to GUI progress (e.g., 20-90%)
                gui_progress_value = 20 + int(percentage_step * 0.7)
                self.progress.emit(gui_progress_value, message)

            # search_and_fetch now returns List[EnrichedItem]
            enriched_items_list: List[EnrichedItem] = search_and_fetch(
                self.query,  # This is query_or_config_path for the backend
                self.logger,
                progress_callback=backend_progress_callback,
                content_type_gui=self.content_type_for_gui if self.content_type_for_gui != 'auto' else None
            )

            status_msg = f"Processed {len(enriched_items_list)} items for '{self.query}'."
            # The logger.enhanced_snippet_data is an internal mechanism for the logger,
            # we now have the full enriched_items_list directly.
            # if hasattr(self.logger, 'enhanced_snippet_data') and self.logger.enhanced_snippet_data:
            #     status_msg += f" Enriched data available for {len(self.logger.enhanced_snippet_data)} details."

            self.progress.emit(100, "Processing complete.")
            self.finished.emit(enriched_items_list, status_msg)

        except Exception as e:
            self.logger.error(f"Error in FetchWorker for query '{self.query}': {e}", exc_info=True)
            user_friendly_message = f"Fetch Error: {type(e).__name__} - {str(e)}. Check logs for detailed error messages."
            self.error.emit(user_friendly_message)


class SaveWorker(QThread):
    finished = Signal(str)  # Emits status message
    error = Signal(str)  # Emits error message

    # MODIFIED: Takes List[EnrichedItem]
    def __init__(self, enriched_items: List[EnrichedItem], directory: str, query_identifier: str,
                 logger_instance: logging.Logger):
        super().__init__()
        self.enriched_items = enriched_items
        self.directory = directory
        self.query_identifier = query_identifier  # Used for base folder name if needed
        self.logger = logger_instance

    def run(self):
        try:
            # Use the new saving function from storage.saver
            save_enriched_items_to_disk(
                enriched_items=self.enriched_items,
                base_output_directory=self.directory,
                query_source_name=self.query_identifier  # Helps in naming the top-level folder for this save operation
            )
            self.finished.emit(f"Successfully saved {len(self.enriched_items)} processed items to {self.directory}.")
        except Exception as e:
            self.logger.error(f"Error saving processed items in SaveWorker: {e}", exc_info=True)
            user_friendly_message = f"Save Error: {type(e).__name__} - {str(e)}. Check logs for details."
            self.error.emit(user_friendly_message)


class EnhancedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(config.APP_NAME)  # Get main app logger
        if not self.logger.handlers:  # Ensure it's configured if main.py didn't run first
            self.logger = setup_logger(name=config.APP_NAME, log_file=config.LOG_FILE_PATH)

        self.setWindowTitle(config.DEFAULT_WINDOW_TITLE)
        self.resize(config.DEFAULT_WINDOW_WIDTH, config.DEFAULT_WINDOW_HEIGHT)

        # MODIFIED: This will store the full EnrichedItem objects from the last fetch
        self.complete_enriched_items_cache: List[EnrichedItem] = []
        # This will store the strings for the "Content Preview" tab
        self.preview_display_strings: List[str] = []

        self._setup_enhanced_ui()
        self.current_content_type_gui_selection = config.DEFAULT_CONTENT_TYPE_FOR_GUI  # From GUI dropdown
        self.on_content_type_change()  # Initialize placeholder text

    def _setup_enhanced_ui(self):
        # ... (UI setup remains largely the same as your provided version) ...
        # Key components:
        # self.content_type_combo, self.mode_combo, self.url_input, self.fetch_button
        # self.progress_bar, self.status_label
        # self.snippets_edit (for Content Preview), self.analysis_edit (for Analysis & Metadata)
        # self.insights_text (for Item Details)
        # self.save_button (for "Save Raw Content"), self.rag_export_button
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Input Group ---
        input_group = QGroupBox("Search Configuration")
        input_layout = QVBoxLayout(input_group)

        if config.SHOW_CONTENT_TYPE_SELECTOR_GUI:
            content_type_layout = QHBoxLayout()
            content_type_layout.addWidget(QLabel("Content Type Hint:"))
            self.content_type_combo = QComboBox()
            for value, display_name in config.LANGUAGE_DISPLAY_NAMES_GUI.items():
                # Check if this content type is generally enabled in config.CONTENT_TYPES
                # or if it's the special 'auto' value.
                if config.CONTENT_TYPES.get(value, True) or value == 'auto':
                    self.content_type_combo.addItem(display_name, value)

            default_idx = self.content_type_combo.findData(config.DEFAULT_CONTENT_TYPE_FOR_GUI)
            if default_idx != -1: self.content_type_combo.setCurrentIndex(default_idx)

            self.content_type_combo.currentTextChanged.connect(self.on_content_type_change)
            content_type_layout.addWidget(self.content_type_combo)
            content_type_layout.addStretch()
            input_layout.addLayout(content_type_layout)

        mode_url_layout = QHBoxLayout()
        mode_url_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Search Query / File Path", "Direct URL"])  # Clarified mode names
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
        mode_url_layout.addWidget(self.mode_combo)

        mode_url_layout.addWidget(QLabel("Input:"))
        self.url_input = QLineEdit()
        # Placeholder text updated by on_mode_change / on_content_type_change
        mode_url_layout.addWidget(self.url_input, 1)

        self.fetch_button = QPushButton("🔍 Fetch & Process")
        self.fetch_button.clicked.connect(self.on_fetch)
        mode_url_layout.addWidget(self.fetch_button)
        input_layout.addLayout(mode_url_layout)
        main_layout.addWidget(input_group)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Results Area (Splitter with Tabs and Details) ---
        results_splitter = QSplitter(Qt.Horizontal)

        # Left side: Tabs for Preview and Analysis
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.results_tabs = QTabWidget()

        self.snippets_edit = QPlainTextEdit()  # For "Content Preview"
        self.snippets_edit.setReadOnly(True)
        self.results_tabs.addTab(self.snippets_edit, "📄 Content Preview")

        self.analysis_edit = QTextEdit()  # For "Analysis & Metadata" - can show HTML
        self.analysis_edit.setReadOnly(True)
        self.results_tabs.addTab(self.analysis_edit, "📊 Analysis & Metadata")

        left_layout.addWidget(self.results_tabs)
        results_splitter.addWidget(left_widget)

        # Right side: Insights/Details for a selected item (future use)
        right_widget = QWidget()  # Placeholder for now
        right_layout = QVBoxLayout(right_widget)
        insights_group = QGroupBox("💡 Item Details (Future)")
        insights_layout = QVBoxLayout(insights_group)
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setPlaceholderText(
            "Select an item from Analysis tab (not yet implemented) or view overall summary in Analysis tab.")
        insights_layout.addWidget(self.insights_text)
        right_layout.addWidget(insights_group)
        results_splitter.addWidget(right_widget)
        results_splitter.setSizes([700, 300])  # Adjust initial sizes

        main_layout.addWidget(results_splitter, 1)  # Give results area more stretch factor

        # --- Bottom Group (Export and Status) ---
        bottom_group = QGroupBox("Actions & Status")
        bottom_outer_layout = QVBoxLayout(bottom_group)

        actions_layout = QHBoxLayout()
        self.save_button = QPushButton("💾 Save Processed Content")  # Renamed button
        self.save_button.setToolTip("Saves the full processed text and structured elements of fetched items to disk.")
        self.save_button.clicked.connect(self.on_save_processed_content)  # Renamed handler
        self.save_button.setEnabled(False)
        actions_layout.addWidget(self.save_button)

        # RAG Export button - currently informational
        self.rag_export_button = QPushButton("ℹ️ RAG Export Info")
        self.rag_export_button.setToolTip(
            "RAG chunks (JSONL/Markdown) are automatically exported by the backend pipeline based on config.")
        self.rag_export_button.clicked.connect(self.on_rag_export_info)
        # self.rag_export_button.setEnabled(False) # It's informational, always enabled
        actions_layout.addWidget(self.rag_export_button)
        actions_layout.addStretch()
        bottom_outer_layout.addLayout(actions_layout)

        self.status_label = QLabel("Ready for modular content scraping.")
        bottom_outer_layout.addWidget(self.status_label)
        main_layout.addWidget(bottom_group)

    def on_content_type_change(self):
        selected_data_value = self.content_type_combo.currentData()
        self.current_content_type_gui_selection = selected_data_value
        self._update_placeholder_text()

    def _update_placeholder_text(self):
        mode = self.mode_combo.currentText()
        type_hint = self.content_type_combo.currentText()  # Display name

        if "URL" in mode:
            self.url_input.setPlaceholderText(f"Enter Direct URL (Content type: {type_hint})")
        else:  # Search Query or File Path
            self.url_input.setPlaceholderText(f"Enter Search Query or YAML Config Path (Content hint: {type_hint})")

        self.snippets_edit.setPlaceholderText(f"Preview of fetched content ({type_hint}) will appear here...")
        self.analysis_edit.setPlaceholderText(f"Analysis & metadata summary ({type_hint}) will appear here...")

    def on_mode_change(self, mode_text):
        self._update_placeholder_text()

    def on_fetch(self):
        query_or_path = self.url_input.text().strip()
        # Determine mode based on ComboBox text, could also use index or data
        mode_text = self.mode_combo.currentText()

        if not query_or_path:
            self.status_label.setText("Please enter a query, URL, or config file path.")
            return

        # Clear previous results
        self.complete_enriched_items_cache = []
        self.preview_display_strings = []
        if hasattr(self.logger, 'enhanced_snippet_data'):  # Reset logger's cache too if it exists
            self.logger.enhanced_snippet_data = []

        self.fetch_button.setEnabled(False)
        self.save_button.setEnabled(False)
        # self.rag_export_button.setEnabled(False) # It's informational

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Initializing processing for: {query_or_path}...")
        self.snippets_edit.setPlainText("")
        self.analysis_edit.setHtml("")
        self.insights_text.setHtml("")

        # Pass the actual value ('html', 'pdf', 'auto') from content_type_combo.currentData()
        gui_content_type_hint = self.content_type_combo.currentData()

        self.fetch_worker = FetchWorker(query_or_path, mode_text, gui_content_type_hint, self.logger)
        self.fetch_worker.progress.connect(self.update_fetch_progress)
        self.fetch_worker.finished.connect(self.handle_fetch_finished)
        self.fetch_worker.error.connect(self.handle_fetch_error)
        self.fetch_worker.start()

    def update_fetch_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    # MODIFIED: Receives List[EnrichedItem]
    def handle_fetch_finished(self, enriched_items: List[EnrichedItem], status_message: str):
        self.complete_enriched_items_cache = enriched_items

        # Generate preview strings for the GUI display
        self.preview_display_strings = []
        for item in self.complete_enriched_items_cache:
            preview = f"Title: {item.title}\nSource: {item.source_url}\n\n"
            if item.primary_text_content:
                preview += item.primary_text_content[:800] + ("..." if len(item.primary_text_content) > 800 else "")
            else:
                preview += "(No primary text content)"

            if item.enriched_structured_elements:
                preview += f"\n\n--- ({len(item.enriched_structured_elements)} structured elements found) ---"
                # Optionally list types of structured elements
                # for elem in item.enriched_structured_elements[:3]:
                #    preview += f"\n  - Type: {elem.get('type', 'unknown')}, Content Snippet: {str(elem.get('content'))[:50]}..."
            self.preview_display_strings.append(preview)

        display_text_for_preview_tab = "\n\n==============================\n\n".join(self.preview_display_strings)
        if not display_text_for_preview_tab and self.url_input.text():  # Check if input was given
            display_text_for_preview_tab = f"No content processed or found for '{self.url_input.text()}'."
        elif not display_text_for_preview_tab:
            display_text_for_preview_tab = "No content processed."

        self.snippets_edit.setPlainText(display_text_for_preview_tab)
        self._display_analysis_summary()  # Update analysis tab based on enriched_items

        self.status_label.setText(status_message)
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if self.complete_enriched_items_cache:  # Enable save if there are items
            self.save_button.setEnabled(True)
        # RAG export info button is always enabled if present

    def _display_analysis_summary(self):
        # Uses self.complete_enriched_items_cache now
        if not self.complete_enriched_items_cache:
            self.analysis_edit.setHtml("<p>No items processed or no metadata to display.</p>")
            return

        summary_html = f"<h3>Overall Processing Summary</h3>"
        summary_html += f"<p><b>Input:</b> {self.url_input.text()}</p>"
        summary_html += f"<p><b>Content Type Hint (GUI):</b> {self.content_type_combo.currentText()}</p>"
        summary_html += f"<p><b>Total Processed Items (displaying summary):</b> {len(self.complete_enriched_items_cache)}</p>"
        summary_html += "<hr>"

        for idx, item in enumerate(self.complete_enriched_items_cache):
            summary_html += f"<h4>Item {idx + 1}: {item.title}</h4>"
            summary_html += f"<ul>"
            summary_html += f"<li><b>Source URL:</b> <a href='{item.source_url}'>{item.source_url}</a></li>"
            summary_html += f"<li><b>Source Type (Hint):</b> {item.source_type}</li>"
            summary_html += f"<li><b>Language (Primary Text):</b> {item.language_of_primary_text or 'N/A'}</li>"
            summary_html += f"<li><b>Categories:</b> {', '.join(item.categories) or 'N/A'}</li>"
            summary_html += f"<li><b>Tags:</b> {', '.join(item.tags[:10]) if item.tags else 'N/A'}</li>"  # Show some tags
            summary_html += f"<li><b>Overall Entities Count:</b> {len(item.overall_entities)}</li>"
            summary_html += f"<li><b>Structured Elements Count:</b> {len(item.enriched_structured_elements)}</li>"

            # Display types of structured elements found
            if item.enriched_structured_elements:
                elem_types = [elem.get('type', 'unknown') for elem in item.enriched_structured_elements]
                elem_type_counts = {t: elem_types.count(t) for t in set(elem_types)}
                summary_html += "<li><b>Structured Element Types:</b><ul>"
                for elem_type, count in elem_type_counts.items():
                    summary_html += f"<li>{elem_type}: {count}</li>"
                summary_html += "</ul></li>"

            # Display a snippet of the metadata_summary for quick glance
            if item.displayable_metadata_summary:
                summary_html += f"<li><b>Quick Metadata Summary:</b><pre>{json.dumps(item.displayable_metadata_summary, indent=2)}</pre></li>"
            summary_html += f"</ul><br>"

        self.analysis_edit.setHtml(summary_html)
        # self.insights_text can be used for more detailed view of a selected item from analysis_edit in future
        # For now, it might just show the same summary or a part of it.
        self.insights_text.setHtml(
            f"<h3>Overall Summary Duplicated</h3> <p>See 'Analysis & Metadata' Tab for full details. Item-specific selection not yet implemented.</p>" + summary_html if len(
                summary_html) < 2000 else "Summary too long for this view, see Analysis tab.")

    def handle_fetch_error(self, error_message):
        self.logger.error(f"GUI received fetch error: {error_message}")
        self.snippets_edit.setPlainText(f"An error occurred during processing:\n\n{error_message}")
        self.status_label.setText("Error during processing. Check logs.")
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_button.setEnabled(False)

    # RENAMED and MODIFIED: on_save_processed_content
    def on_save_processed_content(self):
        if not self.complete_enriched_items_cache:
            self.status_label.setText("No processed content available to save.")
            return

        # Suggest a directory name based on the query/input
        raw_query_text = self.url_input.text().strip()
        # Sanitize query_text to be part of a directory name
        # Replace non-alphanumeric (excluding _, -) with underscore, then take first 30 chars
        sanitized_query_for_dirname = "".join(
            c if c.isalnum() else "_" for c in raw_query_text if c.isalnum() or c in [' ', '_', '-']).strip().replace(
            ' ', '_')
        suggested_dirname_base = f"processed_{sanitized_query_for_dirname[:30]}" if sanitized_query_for_dirname else "processed_content"

        # Allow user to select a base directory for all outputs
        # The saver will create a subfolder within this for this specific save operation.
        base_save_directory = QFileDialog.getExistingDirectory(self, "Select Base Directory to Save Processed Content")
        if not base_save_directory:
            self.status_label.setText("Save cancelled.")
            return

        self.save_worker = SaveWorker(
            self.complete_enriched_items_cache,
            base_save_directory,
            suggested_dirname_base,
            # This will be used by saver to create a sub-folder like "base_save_directory/suggested_dirname_base"
            self.logger
        )
        self.save_worker.finished.connect(self.handle_save_finished)
        self.save_worker.error.connect(self.handle_save_error)

        # Disable buttons during save
        self.save_button.setEnabled(False)
        self.fetch_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress for save
        self.status_label.setText(f"Saving {len(self.complete_enriched_items_cache)} processed items...")
        self.save_worker.start()

    def handle_save_finished(self, status_message):
        self.status_label.setText(status_message)
        self.save_button.setEnabled(True)  # Re-enable after save
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)  # Reset progress bar

    def handle_save_error(self, error_message):
        self.status_label.setText(f"Save Error: {error_message}. Check logs.")
        self.save_button.setEnabled(True)
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

    # RENAMED: on_rag_export_info (was on_rag_export)
    def on_rag_export_info(self):
        self.logger.info(f"GUI RAG Export Info button clicked.")
        # This is informational, actual export happens in backend.
        # Format selection combo was removed as it's backend-controlled now.
        self.status_label.setText(
            f"INFO: RAG Chunks (e.g., JSONL, Markdown) are exported by the backend pipeline based on run configuration."
        )
        self.logger.info(f"To customize RAG export, modify Exporter or use YAML configuration for export path/format.")

    def closeEvent(self, event):
        # Gracefully stop threads if running
        if hasattr(self, 'fetch_worker') and self.fetch_worker.isRunning():
            self.logger.info("Attempting to stop fetch worker on close...")
            self.fetch_worker.quit()  # Request termination
            if not self.fetch_worker.wait(3000):  # Wait up to 3 seconds
                self.logger.warning("Fetch worker did not stop gracefully, terminating.")
                self.fetch_worker.terminate()  # Force terminate
                self.fetch_worker.wait()  # Wait for termination

        if hasattr(self, 'save_worker') and self.save_worker.isRunning():
            self.logger.info("Attempting to stop save worker on close...")
            self.save_worker.quit()
            if not self.save_worker.wait(3000):
                self.logger.warning("Save worker did not stop gracefully, terminating.")
                self.save_worker.terminate()
                self.save_worker.wait()
        event.accept()


if __name__ == '__main__':
    # This block is for running the GUI directly for testing.
    # In production, main.py would typically be the entry point.
    app = QApplication(sys.argv)
    # Ensure a basic logger is set up if running this file directly
    if not logging.getLogger(config.APP_NAME).handlers:
        main_gui_logger = setup_logger(name=config.APP_NAME, log_file="gui_direct_run.log")
        main_gui_logger.info("Running EnhancedMainWindow directly for testing.")

    window = EnhancedMainWindow()
    window.show()
    sys.exit(app.exec())