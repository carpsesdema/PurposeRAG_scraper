# main.py

import sys

from PySide6.QtWidgets import QApplication

import config
from gui.main_window import EnhancedMainWindow
from utils.logger import setup_logger

main_logger = setup_logger(name=config.APP_NAME,
                           log_file=config.LOG_FILE_PATH,
                           console_level_str=config.LOG_LEVEL_CONSOLE,
                           file_level_str=config.LOG_LEVEL_FILE)

if __name__ == "__main__":
    main_logger.info(f"Starting {config.APP_NAME}...")
    app = QApplication(sys.argv)

    if config.STYLESHEET_PATH:
        try:
            with open(config.STYLESHEET_PATH, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
            main_logger.info(f"Global stylesheet '{config.STYLESHEET_PATH}' loaded successfully.")
        except FileNotFoundError:
            main_logger.warning(f"Stylesheet not found at '{config.STYLESHEET_PATH}'. Using default styles.")
        except Exception as e:
            main_logger.error(f"Could not load stylesheet from '{config.STYLESHEET_PATH}': {e}")
    else:
        main_logger.info("No global stylesheet path configured. Application will use default styles.")

    try:
        window = EnhancedMainWindow()
        window.show()
        main_logger.info(f"{config.APP_NAME} GUI started.")
    except Exception as e:
        main_logger.critical(f"Failed to initialize or show the main window: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(app.exec())
