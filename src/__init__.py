import importlib

from beartype import beartype
from beartype.typing import Any
from omegaconf import OmegaConf


@beartype
def resolve_omegaconf_variable(variable_path: str) -> Any:
    # split the string into parts using the dot separator
    parts = variable_path.rsplit(".", 1)

    # get the module name from the first part of the path
    module_name = parts[0]

    # dynamically import the module using the module name
    module = importlib.import_module(module_name)

    # use the imported module to get the requested attribute value
    attribute = getattr(module, parts[1])

    return attribute


def register_custom_omegaconf_resolvers():
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
    OmegaConf.register_new_resolver(
        "resolve_variable", lambda variable_path: resolve_omegaconf_variable(variable_path)
    )
