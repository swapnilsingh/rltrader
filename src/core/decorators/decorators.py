from functools import wraps
from loguru import logger as loguru_logger
import os
import sys
from core.utils.config_loader import load_config

_sink_map = {}

def inject_logger(name_attr="logger_name", level_attr="log_level"):
    def decorator(cls):
        orig_init = cls.__init__

        @wraps(orig_init)
        def wrapped(self, *args, **kwargs):
            logger_name = getattr(self, name_attr, cls.__name__)
            env_key = f"LOG_LEVEL_{logger_name.upper()}"
            default_level = os.getenv("APP_LOG_LEVEL", "WARNING")
            log_level = os.getenv(env_key, getattr(cls, level_attr, default_level)).upper()

            # 🔁 Remove ALL existing sinks once
            if not _sink_map:
                loguru_logger.remove()

            # 🔒 Add one sink per class
            if logger_name not in _sink_map:
                loguru_logger.add(
                    sys.stderr,
                    level=log_level,
                    filter=lambda record: record["extra"].get("source") == logger_name,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " \
                            "<level>{level: <8}</level> | " \
                            "<cyan>{extra[source]}</cyan> | " \
                            "<blue>{function}</blue>:<yellow>{line}</yellow> - " \
                            "<level>{message}</level>"

                )
                _sink_map[logger_name] = True

            bound_logger = loguru_logger.bind(source=logger_name)
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
