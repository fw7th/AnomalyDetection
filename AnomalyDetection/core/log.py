"""
log.py
Logging functionality, saves log files too.
Author: fw7th
Date: 2025-04-25
"""

import logging
import os

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",  
    level=logging.INFO  # Set to DEBUG for detailed logs, ERROR for only failures
)

LOG = logging.getLogger(__name__)  # Best practice: use module name

INFO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "info.log")

ERROR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "error.log")


# Create handlers
info_handler = logging.FileHandler(INFO_DIR)
error_handler = logging.FileHandler(ERROR_DIR)

info_handler.setLevel(logging.INFO)       # Info and above go here
error_handler.setLevel(logging.ERROR)     # Error and above go here

# Create formatters
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Add handlers to the logger
LOG.addHandler(info_handler)
LOG.addHandler(error_handler)
