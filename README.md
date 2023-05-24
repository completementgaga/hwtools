# hwTools

This python package provides tools for the manipulation of handwritten 
text in view of handwritten text recognition, including cursive writing.

The parser module is oriented toward the extraction of the various
lines, words and single characters from a given scanned document.

In view of further statistical treatment, an interactive procedure is 
proposed for the user to control character
extraction and matching with a transcripted version of the text.

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

We highly recommand the use of a virtual environment:
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
[here](https://github.com/completementgaga/copyright-claim/blob/master/sphinx/build/latex/hwtools.pdf).

For the command line version, if you used pip to install the package,

you can run 

```
$ hwtools -h
```

in your terminal and observe
```

```



## Further developments of the project

Feel free to open the discussion through issues and pull requests on the [GitHub repository](https://github.com/completementgaga/hwtools).

Feedback is also welcome by e-mail or by giving the project a star.














