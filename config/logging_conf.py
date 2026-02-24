import logging
import colorlog

def setup_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s',
        log_colors={
            'DEBUG': 'cyan', 'INFO': 'green',
            'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red',
        }
    ))
    logger = logging.getLogger("NVAR")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger