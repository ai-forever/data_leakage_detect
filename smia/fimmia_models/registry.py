import importlib
import os


_MODELS = dict()


def register_all(services_dir):
    # import any Python files in the services_dir directory
    # DO NOT MOVE THIS FUNCTION TO OTHER DIRECTORY
    branches = []
    for branch in os.walk(services_dir):
        branches.append(branch)
    files = []
    for address, _, dir_files in branches:
        for file in dir_files:
            files.append(os.path.join(address, file))
    for path in files:
        path, file = os.path.split(path)
        if (
            not file.startswith('_')
            and not file.startswith('.')
            and file.endswith('.py')
        ):
            module_name = file[:-3]
            module = f"fimmia_v3.fimmia_models.{module_name}"
            # print(f"import module {module}")
            _ = importlib.import_module(f"{module}")


def register_model(cls):
    name = cls.__name__
    _MODELS[name] = cls
    return cls


def rel_dir(file):
    return os.path.split(file)[0]


def get_all_models():
    return _MODELS
