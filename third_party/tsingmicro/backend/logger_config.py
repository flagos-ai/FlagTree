# logger_config.py
import logging
import os
import sys

# Standard mapping: custom number (0~4) -> logging constant
CUSTOM_NUMBER_TO_LOGGING = {
    0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR, 4: logging.CRITICAL
}

# Reverse for validation
LOGGING_TO_CUSTOM_NUMBER = {v: k for k, v in CUSTOM_NUMBER_TO_LOGGING.items()}

# Standard level names for validation
STANDARD_LEVEL_NAMES = {
    name.upper(): getattr(logging, name)
    for name in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
}


def get_log_level_from_env(env_var='TX_LOG_LEVEL', default='info'):
    """
    Read log level from environment variable.
    Supports:
      - String: 'DEBUG', 'info', 'WARNING', etc. (case-insensitive)
      - Number: '0', '1', '2', '3', '4'
    Returns a standard logging level integer (e.g., logging.INFO = 20).
    """
    raw_value = os.getenv(env_var, default).strip()
    if not raw_value:
        raw_value = default

    # Try to interpret as integer (custom 0~4)
    try:
        num = int(raw_value)
        if num in CUSTOM_NUMBER_TO_LOGGING:
            return CUSTOM_NUMBER_TO_LOGGING[num]
        else:
            print(f"Warning: Invalid numeric log level '{num}'. Must be 0~4. Using default '{default}'.")
    except ValueError:
        # Not an integer, treat as string
        level_str = raw_value.upper()
        if level_str in STANDARD_LEVEL_NAMES:
            return STANDARD_LEVEL_NAMES[level_str]
        else:
            print(f"Warning: Invalid log level string '{raw_value}'. "
                  f"Expected one of {list(STANDARD_LEVEL_NAMES.keys())} or 0~4. Using default '{default}'.")

    # Fallback to default
    fallback_level_str = str(default).upper()
    if fallback_level_str in STANDARD_LEVEL_NAMES:
        return STANDARD_LEVEL_NAMES[fallback_level_str]
    elif default.isdigit():
        fallback_num = int(default)
        if fallback_num in CUSTOM_NUMBER_TO_LOGGING:
            return CUSTOM_NUMBER_TO_LOGGING[fallback_num]
    # Final fallback
    return logging.INFO


# Custom log level mapping: standard level names to custom integers (0~4)
CUSTOM_LEVEL_MAP = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}


def log_level_name_to_custom_number(level_name: str) -> int:
    """
    Convert a standard log level name (e.g., 'INFO') to a custom integer (0~4).
    Case-insensitive.
    """
    level_name = level_name.upper()
    if level_name not in CUSTOM_LEVEL_MAP:
        raise ValueError(f"Unsupported log level: {level_name}")
    return CUSTOM_LEVEL_MAP[level_name]


def logger_to_custom_level_number(logger) -> int:
    """
    Get the effective log level of the given logger and convert it to a custom integer (0~4).
    """
    effective_level = logger.getEffectiveLevel()
    level_name = logging.getLevelName(effective_level)

    # Handle non-standard or unrecognized levels
    if not isinstance(level_name, str) or level_name.startswith("Level "):
        raise ValueError(f"Unrecognized log level value: {effective_level}")

    return log_level_name_to_custom_number(level_name)


def log_at_current_level(logger, message):
    current_level = logger.getEffectiveLevel()
    if current_level <= logging.DEBUG:
        logger.debug(message)
    elif current_level <= logging.INFO:
        logger.info(message)
    elif current_level <= logging.WARNING:
        logger.warning(message)
    elif current_level <= logging.ERROR:
        logger.error(message)
    else:
        logger.critical(message)


def setup_logger(name='tsingmicro'):
    """
    Set up and return a unified logger instance.
    Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
    Logs are output to both console and file.
    """
    log_file = f"{name}.log"
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level = get_log_level_from_env()

        # Set logger to lowest level; actual filtering is done by handlers
        logger.setLevel(log_level)

        formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d][%(levelname)s]%(name)s:%(message)s',
                                      datefmt='%Y%m%d %H:%M:%S')

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
