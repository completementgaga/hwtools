# hwTools -- Computer aided handwritten text parsing.
# 
# (C) 2023 Gaël Cousin.
# You may use and distribute this program under the terms of MongoDB's 
# Server Side Public License Version 1, a copy of which you should have received 
# along with this program. Otherwise, see <https://spdx.org/licenses/SSPL-1.0.html>
# or <https://www.mongodb.com/licensing/server-side-public-license>.
# 
# Gaël Cousin can be contacted at gcousin333@gmail.com.

import logging
import os

# logging setup
from .user_config import data_path

log_path = os.path.join(data_path, "hwtools.log")
format_ = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(
    filename=log_path, encoding="utf-8", level=logging.DEBUG, format=format_
)