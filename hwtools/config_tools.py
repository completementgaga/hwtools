# hwTools -- Computer aided handwritten text parsing.
# 
# (C) 2023 Gaël Cousin.
# You may use and distribute this program under the terms of MongoDB's 
# Server Side Public License Version 1, a copy of which you should have received 
# along with this program. Otherwise, see <https://spdx.org/licenses/SSPL-1.0.html>
# or <https://www.mongodb.com/licensing/server-side-public-license>.
# 
# Gaël Cousin can be contacted at gcousin333@gmail.com.

import os
import shutil
import pkgutil

from . import data_manager


def config_dir():
    """Define the path to the configuration directory."""
    # OSwise choice of config directory
    user_config_dir = os.path.join(os.path.expanduser("~"), ".config")
    if os.name == "nt":
        user_config_dir = os.get_env("%LOCALAPPDATA%", user_config_dir)
    user_config_dir = os.path.join(user_config_dir, "hwtools")
    return user_config_dir

def config_file():
    return os.path.join(config_dir(), "hwtools_config.py")


def set_data_path(config_file: str, directory_path: str):
    """Set the user data path in config_file to directory_path.

    Args:
        config_file (str): path to configuration file
        directory_path (str): path to be set as DATA_PATH
    """
    with open(config_file) as f:
        config_lines = f.readlines()
    for i in range(len(config_lines)):
        if "DATA_PATH=" in config_lines[i].replace(' ','').split('#')[0]:
            data_path_index = i
            break
        elif i == len(config_lines) - 1:
            config_lines.append("")
            data_path_index = i + 1
            break
    config_lines[data_path_index] = "DATA_PATH = '" + directory_path + "'\n"
    with open(config_file, 'w') as f:
        f.write("".join(config_lines))

def choose_data_path(user_config_file:str):
    """Have the user put user data path in user_config_file.

    Args:
        user_config_file (str): The file in which DATA_PATH will be set.
    """
    default_data_path = os.path.join(os.path.expanduser("~"), "hwtools")
    data_path = input(
        "Please provide the absolute path to the "
        + "directory where you want the parsed data to be stored.\n"
        + "Simply hit enter if you wish to use the default folder below.\n"
        + default_data_path
    )
    if data_path=='':
        data_path=default_data_path
    data_manager._create_directory(data_path)
    set_data_path(user_config_file, data_path)   


def config_management(reset=False):
    """Ensure the existence/reset of a filled in local config file.

    Args:
        reset (bool, optional): If the DATA_PATH value is to be reset 
            even when already set. Defaults to False.
    """
    user_config_dir = config_dir()
    user_config_file = config_file()

    # The following will allow to reach the data within the package directory
    package_dir = os.path.dirname(pkgutil.resolve_name(__name__).__file__)

    config_template = os.path.join(package_dir, "config_template.py")

    if not os.path.exists(user_config_file) or reset:
        data_manager._create_directory(user_config_dir)
        data_manager.shutil.copyfile(config_template, user_config_file)
        choose_data_path(user_config_file)


