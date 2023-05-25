# hwTools

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
We hope it works fine for most of languages written from left to right
with the Latin (or maybe also Greek) alphabet.

We plan to enlarge this package with algorithms that would exploit enough 
collected data to parse and transcribe new handwritten text automatically,
in a single user based perspective: having the same writing for the 
labeled data and the new text.


## Installation
This package can be installed with pip as follows.

```
$ pip install hwtools
```

We recommand the use of a virtual environment.
For example, if you have Poetry installed, a simple method is 
to create a directory (say with path /my/directory/path)
Then copy the pyproject.toml and poetry.lock to this directory.
Then,

```
$ cd /my/directory/path
$ poetry install
$ poetry shell
$ pip install hwtools
```



## Documentation

A detailed documentation of this package is available on 
[GitHub Pages](https://completementgaga.github.io/hwtools/) --- as a website.

It is also available in pdf format
[here](https://github.com/completementgaga/hwtools/blob/c4b76a9ca9e4bde3c811e00512d8d1f257706534/package_manual.pdf).


### Command line script

For the command line version, if you used pip to install the package,

you can run 

```
$ hwtools -h
```

in your terminal and observe
```
Computer aided handwritten text parsing and matching.
Interactively select from precalculated cut proposals to separate the text in letters and match it with the transcribed version of the text.

(C) 2023 Gaël Cousin.
You may use and distribute this program under the terms of MongoDB's Server Side Public License Version 1, a copy of which you should have received along with this program. Otherwise, see <https://spdx.org/licenses/SSPL-1.0.html> or <https://www.mongodb.com/licensing/server-side-public-license>. Gaël Cousin can be contacted at gcousin333@gmail.com.

options:
  -h, --help       show this help message and exit
  --daltonism, -d  get a color-blind-friendly color scheme in the GUI
  --reset          interactively reset user configuration and exit
```

#### Details for poetry installation
The above paragraph explains how to execute our script from within any (virtual or not) environment. If you installed hwtools within a poetry virtual environment as explained above, to access the script from the ambient
environment you can use the following commands.
```
$ cd /my/directory/path
$ poetry shell
$ hwtools [whatever options]
```
You may as well replace the last two lines by what follows.
```
$ poetry run hwtools [whatever options]
```

## Exiting the script
You can exit the script at any time by hitting Ctrl+C in the terminal.
You might also need to click the script's popping window (see below).

## User configuration
If you first run the hwtools script, you will be asked to provide
a directory for your extracted data. This can be reset using the --reset flag.



## Script usage

After user configuration you will be prompted to input
several informations on the document you wish to parse.

You will have to provide 
 

* a scan of your handwritten document (one page max, png format, for now).
* a typeset transcription of this document
* an estimate for the character's height, thickness and width in your 
handwritten text. 

In the demo_data directory of the github repository, you have the necessary
scan and transcription to perform a first test.


(This is an extract of André Gide, *Les Caves du Vatican*, public domain.)

You also have the file gide_data that shows the format of all the
expected terminal user input (apart for the user configuration of the first run).

For testing needs, I actually used to execute 
```
cat gide_data | hwtools
```
to skip this interaction.

Notice I have the parameters: 
- character width 28
- character height 31
- character thickness 3

I suggest you compress your own scans to reach similar parameter values.
**This is an important step to reduce computational costs.**

### User interface

For each word in the text a window like below will pop up.


| Preview of the GUI|
|:-----------------------------:|
|<img src="https://github.com/completementgaga/hwtools/blob/c4b76a9ca9e4bde3c811e00512d8d1f257706534/screenshots/GUI_preview.png"  width="500em" title="GUI preview">|

Observe the displayed cut proposals, with their round handles. When the window 
appears, they are supposed valid. Click on them to toggle from valid (green) to non-valid(red)
and vice-versa. A --daltonism option is available for color-blind persons.


Below the scanned word extracted from your handwritten text, an extract from
the transcribed version is displayed. It is hopefully close to be the transcription of the displayed word. You are supposed to align perfectly
this transcription using the arrows around it. The arrows on the left modify
the starting point of the extract, the arrows on the right modify its end.

When this is done, close the window to pass to a new word.

If, due to any human or robot failure, you believe you cannot arrange to cut 
correctly the word or to align it with its transcription, you can skip
this word by selecting the 'skip' radio button before closing the window.

### The output
You should find the extracted data organized by author and document title in the
directory you chose in the user configuration step.

For each document you will have a unique folder containing
- a picture for each extracted character, 
-  a picture for each cut (valid or not),
- metadata for the matching of glyphs and
- metadata for the cuts

The skipped words do not contribute to this output.

These data can be retrieved in python with a DataManager from the data_manager
submodule of the hwtools package
and manipulated as MatchedGlyph and CutInfo objects of its parser submodule.
Check [package documentation](https://completementgaga.github.io/hwtools/) for more details.

## Further developments of the project



The most obvious enhancements for this package should be:
 
 - have a more decent GUI. To whom wishes to do this:
    * the structure of the code separates well the UI from the rest.
    * check the Ui abstract class in hwtools.ui_manager
    * you can edit the config file on your machine to use your alternate UI.
    * On Windows it is in your %LOCALAPPDATA%/hwtools
    * On other systems it is in ~/.config/hwtools
 - exploit the collected data for handwriting recognition, I am working on this.

Feel free to open the discussion through issues and pull requests on the [GitHub repository](https://github.com/completementgaga/hwtools).

Feedback is also welcome by e-mail or by giving the project a star.














