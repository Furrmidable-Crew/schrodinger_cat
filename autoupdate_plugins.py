import difflib
import importlib
import inspect
from typing import Callable


def update_plugin(module_name: str) -> Callable:
    """Auto-update the hooks name.

    This decorator allows to keep custom hooks up-to-date with the project.

    As custom hooks need to have the **same name** of original hooks, this decorator looks for the most similar
    methods names in a given module and update the custom hook name to be the same of the closest one.

    Example:

        The decorator should be chained and set as second with respect to the `@hook` decorator. For
        example ::
            @hook()
            @update_plugin("module_name")
            def custom_hook_with_old_name(args):
                 # cool stuff
                 return

    Args:
        module_name: string name of the module where the original hook is defined.

    Returns:
        function with the updated name.

    """
    # Get Absolute Import
    module_path = ".".join(["cat.mad_hatter.core_plugin.hooks", module_name])

    # Import the module
    module = importlib.import_module(module_path)

    # Get all names of the members in hooks module
    functions_list = [f[0] for f in inspect.getmembers(module)]

    def change_name(func):
        # Get the old function name
        current_name = func.__name__

        # Get the most similar name in the given module
        closest_name = difflib.get_close_matches(current_name, functions_list)[0]

        # Set the new name to the hook to be updated
        setattr(func, '__name__', closest_name)

        return func

    return change_name
