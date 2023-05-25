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
---------------------
The ui_manager module
---------------------

Deal with UI aspects.

"""
import os
import shutil
from abc import ABCMeta, abstractmethod

import numpy as np
import skimage
from np_gui import image_annotation
import np_gui.colors
import np_gui.np_clickable_image as np_clickable


from .parser import Word


class Ui(metaclass=ABCMeta):
    """The abstract class for user interfaces."""

    @abstractmethod
    def say(self, text: str) -> None:
        """Inform the user, saying text.

        Args:
            text (str): The text to forward to the user.
        """
        pass

    @abstractmethod
    def get_author(self) -> str:
        """Ask the user to choose an already known author or a new one.

        Returns:
            str: identifier for the author
        """
        pass

    @abstractmethod
    def get_language(self) -> str:
        """Ask the user to provide language details.

        Returns:
            str: the locale code for the language and region.
        """
        pass

    @abstractmethod
    def get_document(self) -> tuple[str, str, str]:
        """Get document data in view of its parsing.

        Returns:
            tuple[str,str,str]: (input_path,transcription_path,doc_title)
        """
        pass

    @abstractmethod
    def get_char_width(self) -> int:
        """Ask for character width, in pixels.

        Returns:
            int: character width
        """
        pass

    @abstractmethod
    def get_char_height(self) -> int:
        """Ask for character height, in pixels.

        Returns:
            int: character height
        """
        pass

    @abstractmethod
    def get_char_thickness(self) -> int:
        """Ask for character thickness, in pixels.

        Returns:
            int: character thickness
        """
        pass

    @abstractmethod
    def choose_cutting_shapes_and_match(
        self,
        word: Word,
        shapes: list[np.ndarray],
        text: str,
        index: int,
        daltonism=False,
    ) -> tuple[list[bool], str]:
        """Select cutting shapes for word and match it to text extract.

        Args:
            word (Word): The Word to be parsed
            shapes (list[np.ndarray]): A list of cutting shape proposals
                to choose from.
            text (str): The typeset version of the text from which Word
                was extracted
            index (int): an estimate of the starting point of the
                typeset version of Word in text.
            daltonism (bool): Decides if a color-blind-friendly color
                scheme is to be used.

        Returns:
            tuple[list[bool], str]: A 2d tuple containing the list of
                booleans that retain which cutting shapes where chosen
                as first entry and the typeset version of word as second
                entry.
        """
        pass


class LocalUi(Ui):
    """A type of user interface, designed for local use of the parser.

    The graphical user interaction is provided using the npGUI package.

    Methods
    -------
    All the methods imposed by the parent abstract class Ui.
    """

    def say(self, text: str) -> None:
        """Inform the user, saying text.

        Args:
            text (str): The text to forward to the user.
        """
        print(text)

    def get_author(self) -> str:
        """Ask the user to choose an already known author or a new one.

        Returns:
            str: identifier for the author
        """
        return input(
            "Indicate a name to identify the author of the"
            + " treated document.\n If other documents of this author were "
            + "already treated you might want to use her/his former identifier"
            + "\n to put in common the informations from all sessions.\n"
        )

    def detect_language(self) -> str:
        """Detects the OS UI language.

        Returns:
            str: the locale code for the language and region.
        """
        if os.name != "posix":
            import locale
            import ctypes

            windll = ctypes.windll.kernel32
            return locale.windows_locale[windll.GetUserDefaultUILanguage()]
        else:
            return os.getenv("LANG")

    def get_language(self) -> str:
        """Ask the user to provide language details.

        Returns:
            str: the locale code for the language and region.
        """

        suggest_format = (
            " provide a language/country code in the format"
            + " fr_FR, en_GB, etc.\n"
            + "This will be used to specify the language of your document.\n"
        )
        try:
            lang = self.detect_language().split(".")[0]
            user_lang = input(
                "We detected the language code "
                + lang
                + " for your computer.\n"
                + "Hit enter if you wish to use this language for"
                + " your document.\n Otherwise, please"
                + suggest_format
            )
            if user_lang:
                lang = user_lang
            return lang
        except:
            return input("Please," + suggest_format)

    def get_document(self) -> tuple[str, str, str]:
        """Get document data in view of its parsing.

        Returns:
            tuple[str,str,str]: (input_path,transcription_path,doc_title)
        """
        scan_path = input(
            "Please, provide the handwritten document path."
            + "For this alpha version it must be a neat prefiltered png file,"
            + "that allows easy relevant conversion to a binary image.\n"
        )
        # + "It can be a png or jpg file containing the page to be treated or"
        # + "a multiple pages pdf document.")
        doc_title = input(
            "Please provide a title for the handwritten document.\n"
        )
        transcription_path = input(
            "Please, provide the path to the transcribed"
            + " version of the document.\n"
        )
        return (scan_path, transcription_path, doc_title)

    def get_char_width(self) -> int:
        """Ask for character width, in pixels.

        Returns:
            int: character width
        """

        val = input(
            "Please enter an estimate of characters' width, in pixels.\n"
            + " Consider the width fo a standard lower case letter, "
            + "like a or o.\n"
        )
        return int(val)

    def get_char_height(self) -> int:
        """Ask for character height, in pixels.

        Returns:
            int: character height
        """

        val = input(
            "Please enter an estimate of characters' height, in pixels.\n"
            + " Consider the height fo a standard upper case letter, "
            + "like A, or a lower case l or t.\n"
        )
        return int(val)

    def get_char_thickness(self) -> int:
        """Ask for character thickness, in pixels.

        Returns:
            int: character thickness
        """

        val = input(
            "Please enter an estimate of characters' thickness, in pixels.\n"
            + "We are meaning: the usual number of consecutive black pixels "
            + "in a character slice.\n"
        )
        return int(val)

    @staticmethod
    def _augmented_region(
        region: np.ndarray, handle_radius: int
    ) -> np.ndarray:
        """Augment a region by adding a circular handle and other pixels.

        Args:
            region (np.ndarray): Binary image with the region as the
                front pixels.
            handle_radius (int): the radius of the added handle.

        Returns:
            np.ndarray: new binary image, with added front pixels.
        """

        support = np.where(region)
        x = int(np.median(support[1]))
        y = int(np.median(support[0]))
        r = np.amax(support[0]) - y
        y_center = y + r + handle_radius
        output = np.zeros(region.shape, dtype="bool")
        output[region] = True
        ellipse = skimage.draw.ellipse(
            y_center, x, handle_radius, handle_radius, shape=region.shape
        )
        output[ellipse] = True
        # We also add some pixels over the region, for more visibility
        output[y - 2 * r : y - r + 1, x] = True
        return output

    def choose_cutting_shapes_and_match(
        self,
        word: Word,
        shapes: list[np.ndarray],
        text: str,
        index: int,
        daltonism=False,
    ) -> tuple[list[bool], str]:
        """Select cutting shapes for word and match it to text extract.

        Args:
            word (Word): The Word to be parsed
            shapes (list[np.ndarray]): A list of cutting shape proposals
                to choose from.
            text (str): The typeset version of the text from which Word
                was extracted
            index (int): an estimate of the starting point of the
                typeset version of Word in text.

        Returns:
            tuple[list[bool], str]: A 2d tuple containing the list of
                booleans that retain which cutting shapes where chosen
                as first entry and the typeset version of word as second
                entry.
        """

        # A background_color option could be added, this is not ready yet.
        background_color = 255 * np.ones((3,), dtype="uint8")
        # Some complex GUI is beeing cooked up out of blocks.
        #  First the top region of the GUI.
        regions = [
            LocalUi._augmented_region(region, word.char_height // 6)
            for region in shapes
        ]
        image_toggles = np_clickable.ImageToggles(
            word.complete_line, regions, daltonism=daltonism
        )
        # The top block  will be the image_toggles block with even
        # white spaces around it.

        # we set here the minimal value of its width
        top_block_width = max(300, image_toggles.shape[1])

        if image_toggles.shape[1] < top_block_width:
            left_white_image = 255 * np.ones(
                (
                    image_toggles.shape[0],
                    (top_block_width - image_toggles.shape[1]) // 2,
                    3,
                ),
                dtype="uint8",
            )
            right_white_image = 255 * np.ones(
                (
                    left_white_image.shape[0],
                    top_block_width
                    - left_white_image.shape[1]
                    - image_toggles.shape[1],
                    3,
                ),
                dtype="uint8",
            )

            top_block = np_clickable.ClickableImage.hstack(
                [left_white_image, image_toggles, right_white_image]
            )
        else:
            top_block = image_toggles
        top_shape = top_block.shape[:2]

        # The bottom block is formed of three regions: a text displayer
        # in the center and two backward/forward blocks on the sides.
        # We build both of these side blocks from two blocks:
        # backward and forward.

        bottom_height = (5 * top_shape[0]) // 10

        # We deal here with the backward/forward blocks
        arrows_width = int(0.1 * top_shape[1])
        arrows_shape = (bottom_height, arrows_width)
        forward_button_image = image_annotation.center_text(
            ">", arrows_shape, background_color=background_color
        )
        backward_button_image = image_annotation.center_text(
            "<", arrows_shape, background_color=background_color
        )

        def backward_left_callback(dic):
            dic["text_start"] = max(0, dic["text_start"] - 1)
            dic["text_slice"] = slice(dic["text_start"], dic["text_stop"])

        def forward_left_callback(dic):
            dic["text_start"] = min(dic["text_start"] + 1, dic["text_stop"])
            dic["text_slice"] = slice(dic["text_start"], dic["text_stop"])

        def backward_right_callback(dic):
            dic["text_stop"] = max(dic["text_start"], dic["text_stop"] - 1)
            dic["text_slice"] = slice(dic["text_start"], dic["text_stop"])

        def forward_right_callback(dic):
            dic["text_stop"] = min(dic["text_stop"] + 1, len(text))
            dic["text_slice"] = slice(dic["text_start"], dic["text_stop"])

        backward_right = np_clickable.Button(
            backward_button_image, backward_right_callback
        )
        backward_left = np_clickable.Button(
            backward_button_image, backward_left_callback
        )
        forward_right = np_clickable.Button(
            forward_button_image, forward_right_callback
        )
        forward_left = np_clickable.Button(
            forward_button_image, forward_left_callback
        )

        text_width = top_shape[1] - 4 * arrows_width
        text_shape = (bottom_height, text_width)

        text_display = np_clickable.SliceDisplayer(
            text,
            text_shape,
            slice_varname="text_slice",
            background_color=background_color,
        )
        bottom_block = np_clickable.ClickableImage.hstack(
            [
                backward_left,
                forward_left,
                text_display,
                backward_right,
                forward_right,
            ]
        )
        # We will actually add a radio button on the line before the 'top'
        # block, to allow the option to skip the word parsing.
        # We call this line 'pretop', since it will be over the top block.
        radio_radius = 10
        radio_button = np_clickable.RadioButton(radio_radius)
        radio_button = radio_button.center_in_shape((40, 25))
        radio_label_shape = (radio_button.shape[0], 70)
        radio_label = image_annotation.center_text("skip:", radio_label_shape)
        pretop_left_margin = np_gui.colors.mono_block(
            (radio_label.shape[0], top_shape[1] - radio_label.shape[1]),
            "white",
        )
        pretop = np_clickable.ClickableImage.hstack(
            [pretop_left_margin, radio_label]
        )
        main_column = np_clickable.ClickableImage.vstack(
            [pretop, top_block, bottom_block]
        )
        radio_bottom_margin = np_gui.colors.mono_block(
            (
                main_column.shape[0] - radio_button.shape[0],
                radio_button.shape[1],
            ),
            "white",
        )
        radio_column = np_clickable.ClickableImage.vstack(
            [radio_button, radio_bottom_margin]
        )

        whole_gui = np_clickable.ClickableImage.hstack(
            [main_column, radio_column]
        )

        whole_gui.vars_dic["text_start"] = index
        whole_gui.vars_dic["text_stop"] = index + len(shapes) + 1
        whole_gui.vars_dic["text_slice"] = slice(
            whole_gui.vars_dic["text_start"], whole_gui.vars_dic["text_stop"]
        )
        return whole_gui.use()
