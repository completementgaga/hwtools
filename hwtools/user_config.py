import sys

from . import config_tools

config_tools.config_management()
sys.path.append(config_tools.config_dir())
from hwtools_config import UI_MODULE_NAME as ui_module_name
from hwtools_config import UI_NAME as ui_name
from hwtools_config import DATA_PATH as data_path

