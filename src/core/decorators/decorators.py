# core/decorators/decorators.py

from functools import wraps
from loguru import logger as loguru_logger
import os
import sys
from core.utils.config_loader import load_config

_loguru_sink_initialized = False

def inject_logger(name_attr="logger_name", level_attr="log_level"):
    def decorator(cls):
        orig_init = cls.__init__

        @wraps(orig_init)
        def wrapped(self, *args, **kwargs):
            global _loguru_sink_initialized

            # Use class name as default logger name
            logger_name = getattr(self, name_attr, cls.__name__)
            env_key = f"LOG_LEVEL_{logger_name.upper()}"
            log_level = os.getenv(env_key, getattr(cls, level_attr, os.getenv("APP_LOG_LEVEL", "INFO"))).upper()

            # Setup Loguru sink once with global config
            if not _loguru_sink_initialized:
                loguru_logger.remove()  # remove default sink
                loguru_logger.add(sys.stderr, level=os.getenv("APP_LOG_LEVEL", "INFO").upper())
                _loguru_sink_initialized = True

            # Bind logger with class-specific source name
            bound_logger = loguru_logger.bind(source=logger_name)
            bound_logger.level(log_level)  # tag level for introspection (optional)

            self.logger = bound_logger
            orig_init(self, *args, **kwargs)

        cls.__init__ = wrapped
        return cls
    return decorator





def inject_config(config_path=None):
    """
    Decorator to inject a YAML config into self.config and as attributes.
    Accepts path like "configs/agent.yaml" or just "agent.yaml".
    """
    def decorator(cls):
        orig_init = cls.__init__

        @wraps(orig_init)
        def wrapped(self, *args, **kwargs):
            normalized = config_path.replace("configs/", "") if config_path else None
            self_config = load_config(normalized) if normalized else {}

            self.config = self_config

            for key, value in self_config.items():
                setattr(self, key, value)

            orig_init(self, *args, **kwargs)

        cls.__init__ = wrapped
        return cls
    return decorator
