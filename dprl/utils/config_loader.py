import sys
sys.path.append(".")
import importlib


def get_config(dataset_code, model_type, task_type):
    """Load the configuration dictionary for the specified dataset."""
    try:
        module_path = f"configs.{task_type}.{model_type}"
        config_module = importlib.import_module(module_path)
        config = getattr(config_module, f"config_{dataset_code}")
        return config
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load configuration for {dataset_code}: {e}")
