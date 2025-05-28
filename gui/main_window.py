import logging  # Ensure logging is imported
import sys

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QPlainTextEdit, QFileDialog,
    QLabel, QComboBox, QProgressBar, QGroupBox, QTextEdit, QTabWidget, QSplitter
)
from storage.saver import save_snippets

import config
from scraper.searcher import search_and_fetch  # , detect_content_type # detect_content_type is now internal to searcher
from utils.logger import setup_logger


class FetchWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(list, str)
    error = Signal(str)

    def __init__(self, query, mode, content_type_for_gui, logger_instance):
        super().__init__()
        self.query = query
        self.mode = mode
        self.content_type_for_gui = content_type_for_gui
        self.logger = logger_instance
        self.total_sources = config.SEARCH_SOURCES_COUNT

    def run(self):
        try:
            snippets = []

            self.progress.emit(10, f"Starting search for: {self.query} (Type: {self.content_type_for_gui})...")

            def backend_progress_callback(message, percentage_step):
                gui_progress_value = 20 + int(percentage_step * 0.7)
                self.progress.emit(gui_progress_value, message)

            snippets = search_and_fetch(
                self.query,
                self.logger,
                progress_callback=backend_progress_callback,
                content_type_gui=self.content_type_for_gui
            )

            status_msg = f"Found {len(snippets)} items for '{self.query}'."
            if hasattr(self.logger, 'enhanced_snippet_data') and self.logger.enhanced_snippet_data:
                status_msg += f" Enriched data available for {len(self.logger.enhanced_snippet_data)} details."

            self.progress.emit(100, "Search complete.")
            self.finished.emit(snippets, status_msg)

        except Exception as e:
            self.logger.error(f"Error in FetchWorker for query '{self.query}': {e}", exc_info=True)
            user_friendly_message = f"Fetch Error: {type(e).__name__} - {str(e)}. Check logs for details."
            self.error.emit(user_friendly_message)


class SaveWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, snippets, directory, content_type, logger_instance):
        super().__init__()
        self.snippets = snippets
        self.directory = directory
        self.content_type = content_type
        self.logger = logger_instance

    def run(self):
        try:
            save_snippets(self.snippets, self.directory)
            self.finished.emit(f"Saved {len(self.snippets)} {self.content_type} snippets to {self.directory}.")
        except Exception as e:
            self.logger.error(f"Error saving snippets in SaveWorker: {e}", exc_info=True)
            user_friendly_message = f"Save Error: {type(e).__name__} - {str(e)}. Check logs for details."
            self.error.emit(user_friendly_message)


class EnhancedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(config.APP_NAME)
        if not self.logger.handlers:
            self.logger = setup_logger(name=config.APP_NAME, log_file=config.LOG_FILE_PATH)

        self.setWindowTitle(config.DEFAULT_WINDOW_TITLE)
        self.resize(config.DEFAULT_WINDOW_WIDTH, config.DEFAULT_WINDOW_HEIGHT)

        self.enhanced_snippet_data_cache = []
        self.snippets_for_display = []

        self._setup_enhanced_ui()
        self.current_content_type = config.DEFAULT_CONTENT_TYPE
        self.on_content_type_change()

    def _setup_enhanced_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        input_group = QGroupBox("Search Configuration")
        input_layout = QVBoxLayout(input_group)

        if config.SHOW_LANGUAGE_SELECTOR:
            content_type_layout = QHBoxLayout()
            content_type_layout.addWidget(QLabel("Content Type:"))
            self.content_type_combo = QComboBox()
            for value, display_name in config.LANGUAGE_DISPLAY_NAMES.items():
                if config.CONTENT_TYPES.get(value, True) or value == 'auto':
                    self.content_type_combo.addItem(display_name, value)

            default_idx = self.content_type_combo.findData(config.DEFAULT_CONTENT_TYPE)
            if default_idx != -1:
                self.content_type_combo.setCurrentIndex(default_idx)
            else:
                first_valid_idx = self.content_type_combo.findData('html')
                if first_valid_idx != -1: self.content_type_combo.setCurrentIndex(first_valid_idx)

            self.content_type_combo.currentTextChanged.connect(self.on_content_type_change)
            content_type_layout.addWidget(self.content_type_combo)
            content_type_layout.addStretch()
            input_layout.addLayout(content_type_layout)

        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Search", "URL"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter search query or seed URL")

        self.fetch_button = QPushButton("🔍 Fetch Content")
        self.fetch_button.clicked.connect(self.on_fetch)

        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(QLabel("Query/URL:"))
        mode_layout.addWidget(self.url_input, 1)
        mode_layout.addWidget(self.fetch_button)
        input_layout.addLayout(mode_layout)

        main_layout.addWidget(input_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        results_splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.results_tabs = QTabWidget()
        self.snippets_edit = QPlainTextEdit()
        self.snippets_edit.setReadOnly(True)
        self.results_tabs.addTab(self.snippets_edit, "📄 Content Preview")

        if config.CODE_CATEGORIZATION_ENABLED or True:
            self.analysis_edit = QTextEdit()
            self.analysis_edit.setReadOnly(True)
            self.results_tabs.addTab(self.analysis_edit, "📊 Analysis & Metadata")

        left_layout.addWidget(self.results_tabs)
        results_splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        insights_group = QGroupBox("💡 Item Details")
        insights_layout = QVBoxLayout(insights_group)
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        insights_layout.addWidget(self.insights_text)
        right_layout.addWidget(insights_group)
        results_splitter.addWidget(right_widget)
        results_splitter.setSizes([700, 200])

        main_layout.addWidget(results_splitter)

        bottom_group = QGroupBox("Export Options")
        bottom_layout = QVBoxLayout(bottom_group)

        standard_layout = QHBoxLayout()
        self.save_button = QPushButton("💾 Save Raw Content")
        self.save_button.clicked.connect(self.on_save_raw_content)
        self.save_button.setEnabled(False)
        standard_layout.addWidget(self.save_button)
        standard_layout.addStretch()
        bottom_layout.addLayout(standard_layout)

        if config.EMBEDDING_RAG_EXPORT_ENABLED:
            enhanced_layout = QHBoxLayout()
            self.export_format_combo = QComboBox()
            self.export_format_combo.addItems(["jsonl", "markdown", "yaml"])

            self.rag_export_button = QPushButton("📦 Export RAG Chunks")
            self.rag_export_button.clicked.connect(self.on_rag_export)
            self.rag_export_button.setEnabled(False)

            enhanced_layout.addWidget(QLabel("Format:"))
            enhanced_layout.addWidget(self.export_format_combo)
            enhanced_layout.addWidget(self.rag_export_button)
            enhanced_layout.addStretch()
            bottom_layout.addLayout(enhanced_layout)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready for modular content scraping.")
        status_layout.addWidget(self.status_label, 1)
        bottom_layout.addLayout(status_layout)
        main_layout.addWidget(bottom_group)

    def on_content_type_change(self):
        selected_data_value = self.content_type_combo.currentData()
        self.current_content_type = selected_data_value
        self._update_placeholder_text()

    def _update_placeholder_text(self):
        mode = self.mode_combo.currentText()
        if mode == "URL":
            self.url_input.setPlaceholderText(f"Enter URL for {self.current_content_type} content...")
        else:
            self.url_input.setPlaceholderText(f"Enter search query for {self.current_content_type}...")

        self.snippets_edit.setPlaceholderText(f"Fetched {self.current_content_type} content will appear here...")
        if hasattr(self, 'analysis_edit'):
            self.analysis_edit.setPlaceholderText(f"Analysis of {self.current_content_type} content...")
        if hasattr(self, 'insights_text'):
            self.insights_text.setPlaceholderText(f"Details for selected {self.current_content_type} item...")

    def on_mode_change(self, mode):
        self._update_placeholder_text()

    def on_fetch(self):
        query = self.url_input.text().strip()
        mode = self.mode_combo.currentText()

        if not query:
            self.status_label.setText("Please enter a query or URL.")
            return

        if hasattr(self.logger, 'enhanced_snippet_data'):  # Check before assigning
            self.logger.enhanced_snippet_data = []
        self.enhanced_snippet_data_cache = []
        self.snippets_for_display = []

        self.fetch_button.setEnabled(False)
        self.save_button.setEnabled(False)
        if hasattr(self, 'rag_export_button'): self.rag_export_button.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Initializing search for: {query} (Type: {self.current_content_type})...")
        self.snippets_edit.setPlainText("")
        if hasattr(self, 'analysis_edit'): self.analysis_edit.setHtml("")
        if hasattr(self, 'insights_text'): self.insights_text.setHtml("")

        self.fetch_worker = FetchWorker(query, mode, self.current_content_type, self.logger)
        self.fetch_worker.progress.connect(self.update_fetch_progress)
        self.fetch_worker.finished.connect(self.handle_fetch_finished)
        self.fetch_worker.error.connect(self.handle_fetch_error)
        self.fetch_worker.start()

    def update_fetch_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def handle_fetch_finished(self, raw_snippets_list, status_message):
        self.snippets_for_display = raw_snippets_list

        if hasattr(self.logger, 'enhanced_snippet_data'):
            self.enhanced_snippet_data_cache = self.logger.enhanced_snippet_data

        display_text = "\n\n--------------------\n\n".join(self.snippets_for_display)
        if not display_text:
            display_text = f"No {self.current_content_type} content found for '{self.url_input.text()}'."
        self.snippets_edit.setPlainText(display_text)

        self._display_analysis_and_insights()

        self.status_label.setText(status_message)
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if self.snippets_for_display:
            self.save_button.setEnabled(True)
        if self.enhanced_snippet_data_cache and hasattr(self, 'rag_export_button'):
            self.rag_export_button.setEnabled(True)

    def _display_analysis_and_insights(self):
        if hasattr(self, 'analysis_edit') and self.enhanced_snippet_data_cache:
            summary_html = f"<h3>Analysis Summary</h3>"
            summary_html += f"<p><b>Query:</b> {self.url_input.text()}</p>"
            summary_html += f"<p><b>Content Type:</b> {self.current_content_type}</p>"
            summary_html += f"<p><b>Total Raw Items Processed by GUI:</b> {len(self.snippets_for_display)}</p>"
            summary_html += f"<p><b>Total Enriched Detail Sets:</b> {len(self.enhanced_snippet_data_cache)}</p>"

            if self.enhanced_snippet_data_cache:
                first_item_details = self.enhanced_snippet_data_cache[0]
                summary_html += "<h4>Example First Item Details:</h4><ul>"
                for key, value in first_item_details.items():
                    if key == 'code' or key == 'text_content':
                        summary_html += f"<li><b>{key}:</b> (Content previewed separately)</li>"
                    elif isinstance(value, dict):
                        summary_html += f"<li><b>{key}:</b> {len(value)} sub-items</li>"
                    elif isinstance(value, list):
                        summary_html += f"<li><b>{key}:</b> {len(value)} items</li>"
                    else:
                        summary_html += f"<li><b>{key}:</b> {str(value)[:100]}</li>"
                summary_html += "</ul>"

            self.analysis_edit.setHtml(summary_html)

        if hasattr(self, 'insights_text'):
            self.insights_text.setPlaceholderText(
                "Select an item from 'Analysis & Metadata' or future list view to see details."
            )
            self.insights_text.setHtml("")

    def handle_fetch_error(self, error_message):
        self.logger.error(f"GUI received fetch error: {error_message}")
        self.snippets_edit.setPlainText(f"An error occurred during fetch:\n\n{error_message}")
        self.status_label.setText("Error during fetch. Check logs.")
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_button.setEnabled(False)
        if hasattr(self, 'rag_export_button'): self.rag_export_button.setEnabled(False)

    def on_save_raw_content(self):
        if not self.snippets_for_display:
            self.status_label.setText("No content to save.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Raw Content")
        if not directory:
            self.status_label.setText("Save cancelled.")
            return

        self.save_worker = SaveWorker(self.snippets_for_display, directory, self.current_content_type, self.logger)
        self.save_worker.finished.connect(self.handle_save_finished)
        self.save_worker.error.connect(self.handle_save_error)

        self.save_button.setEnabled(False)
        self.fetch_button.setEnabled(False)
        if hasattr(self, 'rag_export_button'): self.rag_export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText(f"Saving {len(self.snippets_for_display)} raw items...")
        self.save_worker.start()

    def handle_save_finished(self, status_message):
        self.status_label.setText(status_message)
        self.save_button.setEnabled(True)
        self.fetch_button.setEnabled(True)
        if self.enhanced_snippet_data_cache and hasattr(self, 'rag_export_button'):
            self.rag_export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

    def handle_save_error(self, error_message):
        self.status_label.setText(f"Save Error: {error_message}. Check logs.")
        self.save_button.setEnabled(True)
        self.fetch_button.setEnabled(True)
        if self.enhanced_snippet_data_cache and hasattr(self, 'rag_export_button'):
            self.rag_export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

    def on_rag_export(self):
        if not self.enhanced_snippet_data_cache:
            self.status_label.setText("No enriched data available to generate RAG export.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory for RAG Export")
        if not directory:
            self.status_label.setText("RAG Export cancelled.")
            return

        export_format = self.export_format_combo.currentText()

        # The new backend `run_pipeline` handles export directly.
        # This GUI button could, in the future, trigger a re-export or export cached RAGOutputItems.
        # For now, we inform the user that the backend handles the main export.
        self.logger.info(f"GUI RAG Export button clicked for format: {export_format}.")
        self.status_label.setText(
            f"RAG export ({export_format}) is primarily handled by the backend pipeline during fetch.")
        self.logger.info(f"To customize RAG export, modify the Exporter in the backend pipeline or its configuration.")

        # As a fallback, if you want the GUI to save the *enriched details* (not full RAG chunks):
        # from pathlib import Path
        # output_file_path = Path(directory) / f"gui_enriched_export_{self.url_input.text().replace(' ','_')}.{export_format}"
        # try:
        #     if export_format == "jsonl":
        #         with open(output_file_path, 'w', encoding='utf-8') as f:
        #             for enriched_detail_set in self.enhanced_snippet_data_cache:
        #                 f.write(json.dumps(enriched_detail_set) + '\n')
        #         self.status_label.setText(f"Exported enriched details (not full RAG chunks) to {output_file_path}")
        #     else:
        #         self.status_label.setText(f"Format {export_format} for direct GUI enriched data export not implemented.")
        # except Exception as e:
        #     self.logger.error(f"Error during GUI enriched data export: {e}", exc_info=True)
        #     self.status_label.setText(f"GUI enriched data export failed: {e}")

    def closeEvent(self, event):
        if hasattr(self, 'fetch_worker') and self.fetch_worker.isRunning():
            self.logger.info("Attempting to stop fetch worker...")
            self.fetch_worker.quit()
            if not self.fetch_worker.wait(3000):
                self.logger.warning("Fetch worker did not stop gracefully, terminating.")
                self.fetch_worker.terminate()
                self.fetch_worker.wait()

        if hasattr(self, 'save_worker') and self.save_worker.isRunning():
            self.save_worker.quit()
            self.save_worker.wait(3000)

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EnhancedMainWindow()
    window.show()
    sys.exit(app.exec())
