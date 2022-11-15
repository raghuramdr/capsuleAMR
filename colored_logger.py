import logging
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import Queue
from typing import Optional


class ColoredFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    purple = "\x1b[1;35m"
    blink_red = "\x1b[5m\x1b[1;31m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO:  purple + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: blink_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record) 


class LoggerContextManager:
    """
    Wrap the logger inside a context manager to close the QueueListener after it's no longer needed.
    """
    def __init__(self, log_path: Optional[Path] = None):
        """
        Configure the Logger inside the ContextManager.

        Parameters
        ----------
        log_path : Path | None
            Directory where to write log files. If None then use the current working directory.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Set file to save logs
        log_folder = log_path if log_path else Path()
        log_folder.mkdir(exist_ok=True)
        log_file = log_folder.joinpath(f"download_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
        # Create and configure handler to write logs into file
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)

        # Create Queue to write logs in separate Thread into the file to avoid block the async code
        log_queue = Queue()
        queue_handler = logging.handlers.QueueHandler(log_queue)
        self.queue_listener = QueueListener(log_queue, file_handler)
        self.logger.addHandler(queue_handler)

        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter())
        self.logger.addHandler(ch)

    def __enter__(self) -> logging.Logger:
        self.queue_listener.start()
        return self.logger

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.queue_listener.stop()
