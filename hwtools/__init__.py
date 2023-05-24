"""
-------------------------
    The hwTools package
-------------------------

This package provides tools for the manipulation of handwritten text in
view of handwritten text recognition, including cursive writing.

The parser module is oriented toward the extraction of the various
lines, words and single characters from a given scanned document.

In view of further statistical treatment, an interactive procedure is 
proposed for the user to control character
extraction and matching with a transcripted version of the text.

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
