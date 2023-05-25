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
-------------------------------
the training_data_script module 
-------------------------------

Provide the script for interactive parsing.

"""
import os
import sys
import argparse
import importlib

import numpy as np

from . import ui_manager
from . import parser
from . import data_manager
from . import config_tools
from . import user_config


from . import log_config

logger = log_config.logging.getLogger("hwtools.training_data_script")

Chosen_Ui = getattr(
    importlib.import_module(user_config.ui_module_name),
    user_config.ui_name,
)
data_path = user_config.data_path


def treat_word(
    my_ui: ui_manager.Ui,
    my_data_manager: data_manager.DataManager,
    text: str,
    word: parser.Word,
    start_index: int,
    max_errors_count: int,
    daltonism: bool,
) -> int:
    """Parse interactively a Word.

    This function is mainly meant to appear in a loop, when we parse
    all the Words (parser.Word) of a page. By parsing, we mean:

        - choosing relevant cuts in the word to separate its various
          letters, these cuts are chosen among a certain number of
          cut proposals

        - matching these letters with the transcribed text,

        - storing the corresponding parser.MatchedGlyphs on disk,

        - storing the information of which cuts were relevant, together
          with  an image extract to retain in which context they
          appeared (parser.CutInfos).


    Args:
        my_ui (Ui): The UI being used.
        my_data_manager: The data manager being used.
        text (str): The compacted transcribed version of the text being
             parsed. Compacted means spaces and lineskips were removed.
        word (Word): The Word to be parsed.
        start_index (int): the index of text at which the
            word transcription is supposed to start.
        max_errors_count (int): The maximum number of parsing errors
            before we decide to do nothing for (skip) this word.
        daltonism (bool): Defaults to False. If False the default color
            scheme is used, the ability to distinguish usual green and red
            is necessary. If set to True a color-blind-friendly color
            scheme is used.

    Returns:
        int: The start index for the next Word in the loop.
    """

    shapes, cut_positions = word.cutting_shapes()
    can_match = False
    errors_count = 0
    skipped = False
    while (not can_match) and errors_count <= max_errors_count:
        dic = my_ui.choose_cutting_shapes_and_match(
            word, shapes, text, start_index, daltonism=daltonism
        )

        # Check if the skipping option was used.
        if dic["radio_toggle"][0]:
            my_ui.say("Word skipped by the user.")
            break

        selected_shapes = [
            shape for i, shape in enumerate(shapes) if dic["toggles"][i]
        ]

        cut_word = word.cut(selected_shapes)
        transcribed_word = text[dic["text_start"] : dic["text_stop"]]
        my_matcher = parser.WordMatcher(cut_word, transcribed_word)
        try:
            matched_glyphs = my_matcher.match()
            can_match = True
        except:
            if errors_count < max_errors_count:
                my_ui.say(
                    "It seems "
                    + "there is an issue with the last word, the numbers of"
                    + " characters for the transcribed and handwritten"
                    + " texts do not match.\n"
                    + "Please check again this word."
                )
            else:
                my_ui.say("Still a matching error, passing this word.")
            errors_count += 1
        if can_match:
            # We also will need the CutInfo for  each proposed
            # cutting shape.
            my_cut_parser = parser.CutParser(
                word, shapes, cut_positions, dic["toggles"]
            )
            cut_infos = my_cut_parser.get_cut_infos()

            # Store the parsing data
            my_data_manager.store_data(matched_glyphs, cut_infos)

    # Returning the start index for the the next word.
    return dic["text_stop"]


def main():
    """Offer the terminal command."""
    argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        prog="hwtools",
        description="Computer aided handwritten text parsing and matching.\n"
        + "Interactively select from precalculated cut proposals to "
        + "separate the text in letters and match it with the transcribed "
        + "version of the text.\n\n"
        + "(C) 2023 Gaël Cousin.\n"
        + "You may use and distribute this program under the terms of MongoDB's "
        + "Server Side Public License Version 1, a copy of which you should "
        + "have received along with this program. "
        + "Otherwise, see <https://spdx.org/licenses/SSPL-1.0.html> "
        + "or <https://www.mongodb.com/licensing/server-side-public-license>. "
        + "Gaël Cousin can be contacted at gcousin333@gmail.com.",
    )
    argparser.add_argument(
        "--daltonism",
        "-d",
        help="get a "
        + "color-blind-friendly color scheme in the GUI",
        action="store_true",
        default=False,
    )
    argparser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="interactively reset user configuration and exit",
    )
    args = argparser.parse_args()
    if args.reset:
        config_tools.config_management(reset=True)
        sys.exit(
            "Now quitting, please start the script again to use the new configuration."
        )

    daltonism = args.daltonism

    my_ui = Chosen_Ui()

    author = my_ui.get_author()

    (scan_path, transcription_path, doc_title) = my_ui.get_document()
    storing_directory = data_manager.cook_storing_directory(
        data_path, author, doc_title
    )
    my_data_manager = data_manager.DataManager(storing_directory)
    my_data_manager.store_transcription(transcription_path)
    my_data_manager.store_scan(scan_path)

    char_width = my_ui.get_char_width()
    char_height = my_ui.get_char_height()
    char_thickness = my_ui.get_char_thickness()

    # One could wish to exploit language information to tune parameters.
    # lang = my_ui.get_language()

    text = my_data_manager.compacted_text

    page = parser.Page(
        my_data_manager.image_from_scan(0),
        char_height,
        char_thickness,
        char_width,
    )
    logger.info("Working with the scan " + str(scan_path))
    lines = page.lines()
    logger.info("The number of detected lines is: " + str(page.lines_number()))
    logger.info("The line heights are: " + str(page.line_heights()))

    # main loop
    my_data_manager.set_data_counters()
    start_index = 0
    for line in lines:
        for word in line.words():
            start_index = treat_word(
                my_ui,
                my_data_manager,
                text,
                word,
                start_index,
                2,
                daltonism=daltonism,
            )

    my_ui.say("Finished!")


if __name__ == "__main__":
    main()
