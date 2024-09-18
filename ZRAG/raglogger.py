import logging
def setup_logger(log_file: str = "app.log") -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    if not logger.hasHandlers():
        logger.addHandler(handler)
    
    return logger
