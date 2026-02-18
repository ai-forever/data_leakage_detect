import importlib
from pathlib import Path


_MODELS = dict()


def register_all(services_dir):
    # import any Python files in the services_dir directory
    # DO NOT MOVE THIS FUNCTION TO OTHER DIRECTORY
    services_path = Path(services_dir)
    files = []
    for file_path in services_path.rglob("*.py"):
        if (
            not file_path.name.startswith("_")
            and not file_path.name.startswith(".")
        ):
            files.append(file_path)
    for path in files:
        file = path.name
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and file.endswith(".py")
        ):
            module_name = file[:-3]
            module = f"fimmia.fimmia_models.{module_name}"
            # print(f"import module {module}")
            _ = importlib.import_module(f"{module}")


def register_model(cls):
    name = cls.__name__
    _MODELS[name] = cls
    return cls


def rel_dir(file):
    return str(Path(file).parent)


def get_all_models():
    return _MODELS
