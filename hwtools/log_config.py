import logging
import os

# logging setup
from .user_config import data_path

log_path = os.path.join(data_path, "hwtools.log")
format_ = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(
    filename=log_path, encoding="utf-8", level=logging.DEBUG, format=format_
)