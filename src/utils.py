# src/utils.py

import logging
import os

def setup_logger(name='health_logger', log_file='health.log'):
    """Setup a logger"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def create_dir_if_not_exists(path: str):
    """Create directory if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)
