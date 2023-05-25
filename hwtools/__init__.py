# hwTools -- Computer aided handwritten text parsing.
# 
# (C) 2023 Gaël Cousin.
# You may use and distribute this program under the terms of MongoDB's 
# Server Side Public License Version 1, a copy of which you should have received 
# along with this program. Otherwise, see <https://spdx.org/licenses/SSPL-1.0.html>
# or <https://www.mongodb.com/licensing/server-side-public-license>.
# 
# Gaël Cousin can be contacted at gcousin333@gmail.com.

"""
-------------------------
    The hwTools package
-------------------------

This python package provides tools for the manipulation of handwritten 
text in view of handwritten text recognition, including cursive writing.

The parser module is oriented toward the extraction of the various
lines, words and single characters from a given scanned document.

In view of further statistical treatment, an interactive procedure is 
proposed for the user to control character
extraction and matching with a transcribed version of the text.

It is the main procedure of the training_data_script module.

The ui_manager module deals with UI aspects of this interaction and 
the data_manager module deals with the I/O aspects.

The parser module is conceived with the cultural biases of a french author.
We hope it works fine for most of western languages.

We plan to enlarge this package with algorithms that would exploit enough 
collected data to parse and transcribe new handwritten text automatically,
in a single user based perspective: having the same writing for the 
labeled data and the new text.


"""

from . import parser
from . import data_manager
from . import ui_manager
# from . import training_data_script