# core/decorators/decorators.py

from functools import wraps
from loguru import logger as loguru_logger

from core.utils.config_loader import load_config

def inject_logger(name_attr="logger_name", level_attr="log_level"):
    def decorator(cls):
        orig_init = cls.__init__

        @wraps(orig_init)
        def wrapped(self, *args, **kwargs):
            # Determine logger name and level from class attributes or defaults
            logger_name = getattr(self, name_attr, cls.__name__)
            log_level = getattr(cls, level_attr, "INFO").upper()

            # Bind a contextual Loguru logger
            bound_logger = loguru_logger.bind(source=logger_name)
            self.logger = bound_logger

            # Optionally set level (globally, if needed)
            # loguru_logger.remove()
            # loguru_logger.add(sys.stderr, level=log_level)

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
