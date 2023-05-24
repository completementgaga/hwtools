# hwTools -- Computer aided handwritten text parsing.
# 
# (C) 2023 Gaël Cousin.
# You may use and distribute this program under the terms of MongoDB's 
# Server Side Public License Version 1, a copy of which you should have received 
# along with this program. Otherwise, see <https://spdx.org/licenses/SSPL-1.0.html>
# or <https://www.mongodb.com/licensing/server-side-public-license>.
# 
# Gaël Cousin can be contacted at gcousin333@gmail.com.



# USER CONFIGURATION FILE FO THE HWTOOLS PACKAGE

# Edit the two lines below if you want to use/test an alternate UI class.
# Check the documentation/code of the ui_manager submodule, in particular
# its Ui abstract class that specifies the necessary methods for an UI:
# your UI class should be a subclass of this abstract class.
# if necessary:
# sys.path.append(path_to_find_your_UI_MODULE)
UI_MODULE_NAME= 'hwtools.ui_manager'
UI_NAME = 'LocalUi'



# The user might wish to change this DATA_PATH variable.
# It is the path where the parsed text data will be stored, together with
# logs.
DATA_PATH = 'to be defined'

