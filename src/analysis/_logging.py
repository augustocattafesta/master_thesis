import sys
import loguru
from loguru import logger

from analysis.context import Context
from analysis.config import TaskType

def setup_logging() -> "loguru.logger":
    """Set up logging configuration for the analysis module."""
    logger.remove()

    fmt = (
        "<blue>{elapsed}</blue> | "
        "<level>[{level: <8}]</level> | "
        "<blue>{message}</blue> "
        )
    
    logger.add(sys.stderr, format=fmt, level="DEBUG", colorize=True)

    return logger


def log_task(func: callable):
    """Decorator to log the execution of a task function."""
    def wrapper(context: Context, task: TaskType) -> Context:
        logger.info(f"Starting task <yellow>{task.task}</yellow>")
        result = func(context, task)
        logger.info(f"Finished task <green>{task.task}</green>")
        return result
    return wrapper


logger = setup_logging()