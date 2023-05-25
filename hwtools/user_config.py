# hwTools -- Computer aided handwritten text parsing.
# 
# (C) 2023 Gaël Cousin.
# You may use and distribute this program under the terms of MongoDB's 
# Server Side Public License Version 1, a copy of which you should have received 
# along with this program. Otherwise, see <https://spdx.org/licenses/SSPL-1.0.html>
# or <https://www.mongodb.com/licensing/server-side-public-license>.
# 
# Gaël Cousin can be contacted at gcousin333@gmail.com.

"""Provide user configuration constants."""
import sys

from . import config_tools

config_tools.config_management()
sys.path.append(config_tools.config_dir())
from hwtools_config import UI_MODULE_NAME as ui_module_name
from hwtools_config import UI_NAME as ui_name
from hwtools_config import DATA_PATH as data_path

