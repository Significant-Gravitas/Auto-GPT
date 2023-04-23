"""Handles loading of plugins."""

import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from zipimport import zipimporter

from auto_gpt_plugin_template import AutoGPTPluginTemplate

from autogpt.config import Config


def inspect_zip_for_modules(zip_path: str, debug: bool = False) -> list[str]:
    """
    Inspect a zipfile for a modules.

    Args:
        zip_path (str): Path to the zipfile.
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        list[str]: The list of module names found or empty list if none were found.
    """
    result = []
    with zipfile.ZipFile(zip_path, "r") as zfile:
        for name in zfile.namelist():
            if name.endswith("__init__.py"):
                if debug:
                    print(f"Found module '{name}' in the zipfile at: {name}")
                result.append(name)
    if debug and len(result) == 0:
        print(f"Module '__init__.py' not found in the zipfile @ {zip_path}.")
    return result


def scan_plugins(cfg: Config, debug: bool = False) -> List[AutoGPTPluginTemplate]:
    """Scan the plugins directory for plugins and loads them.

    Args:
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        List[Tuple[str, Path]]: List of plugins.
    """
    loaded_plugins = []
    # Generic plugins
    plugins_path_path = Path(cfg.plugins_dir)
    for plugin in plugins_path_path.glob("*.zip"):
        if moduleList := inspect_zip_for_modules(str(plugin), debug):
            for module in moduleList:
                plugin = Path(plugin)
                module = Path(module)
                if debug:
                    print(f"Plugin: {plugin} Module: {module}")
                zipped_package = zipimporter(str(plugin))
                zipped_module = zipped_package.load_module(str(module.parent))
                for key in dir(zipped_module):
                    if key.startswith("__"):
                        continue
                    a_module = getattr(zipped_module, key)
                    a_keys = dir(a_module)
                    if (
                        "_abc_impl" in a_keys
                        and a_module.__name__ != "AutoGPTPluginTemplate"
                        and denylist_allowlist_check(a_module.__name__, cfg)
                    ):
                        loaded_plugins.append(a_module())
    if loaded_plugins:
        print(f"\nPlugins found: {len(loaded_plugins)}\n" "--------------------")
    for plugin in loaded_plugins:
        print(f"{plugin._name}: {plugin._version} - {plugin._description}")
    return loaded_plugins


def denylist_allowlist_check(plugin_name: str, cfg: Config) -> bool:
    """Check if the plugin is in the allowlist or denylist.

    Args:
        plugin_name (str): Name of the plugin.
        cfg (Config): Config object.

    Returns:
        True or False
    """
    if plugin_name in cfg.plugins_denylist:
        return False
    if plugin_name in cfg.plugins_allowlist:
        return True
    ack = input(
        f"WARNNG Plugin {plugin_name} found. But not in the"
        " allowlist... Load? (y/n): "
    )
    return ack.lower() == "y"
