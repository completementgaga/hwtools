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
The data_manager module
-------------------------

Manage data input and output."""

from __future__ import annotations
import os
import sys
import shutil
import csv
import datetime
from collections import namedtuple

import numpy as np
import skimage

from . import parser
from . import config_tools



def _create_directory(directory:str) -> None:
    """Create local directory if it does not exist.

    Args:
        directory (str): Desired absolute path to the directory.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def cook_storing_directory(data_path, author: str, doc_title: str) -> str:
    """Cook up a storing directory path from author and doc_title.

    It is a preparatory function, to define DataManagers.

    Args:
        data_path (str): application's user data directory.
        author (str): The parsed document's author.
        doc_title (str): The desired title for the document.


    Returns:
        str: An absolute path.
    """
    timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return os.path.join(data_path, author, doc_title + "_" + timestamp)


# We could have an abstract class for data managers
class DataManager:
    """The class that stores and retrieves parsed data and documents."""

    # The extension used for image files in store_images
    image_ext: str = ".tif"

    # a parameter of the store_image and retrieve_image methods
    contrast_factor: int = 10 * 256

    # name used for scanned documents
    scan_name: str = "scan"

    # name used for transcribed documents
    transcription_name: str = "transcription"

    def __init__(self, storing_directory: str, binary_extracts: bool =True) -> None:
        """Instantiate DataManager and create its directory.

        Args:
            storing_directory (str): The absolute path to the directory
                where the data manager will store its information.
            binary_extracts (bool): Defaults to True. If True all the
            stored text extracts will be binary images. Otherwise,
            the labeling that separates the basic_line and the various
            satellites is preserved, but the images stored on disk are 
            less human readable.
        """
        self.directory = storing_directory
        _create_directory(self.directory)
        self.binary_extracts=binary_extracts

    def store_file(self, source: str, rel_target_path: str) -> str:
        """Store source file to rel_target_path within self.directory.

        Args:
            source (str): The absolute path to the source file.
            rel_target_path (str): The target path, relative to
                self.directory.


        Returns:
            str: Absolute path to stored file.
        """
        abs_target_path = os.sep.join([self.directory, rel_target_path])
        shutil.copyfile(source, abs_target_path)
        return abs_target_path

    def store_scan(self, source: str) -> None:
        """Store scanned document in self.directory.

        Args:
            source (str): absolute path to the document.
        """
        ext = os.path.splitext(source)[-1]
        self.scan_path = self.store_file(source, self.scan_name + ext)

    def store_transcription(self, source: str) -> None:
        """Store transcribed document in self.directory.

        Args:
            source (str): absolute path to the transcribed document.
        """
        ext = os.path.splitext(source)[-1]
        self.transcription_path = self.store_file(
            source, self.transcription_name + ext
        )

    def image_from_scan(self, page_number: int = 0) -> np.ndarray:
        """Return given page of the scanned document as a numpy 2darray.

        In this first version the scan is supposed to have a unique page,
        so that only  0 is a meaningful input.

        Args:
            page_number (int, optional): The page number. Defaults to 0.
        """
        return skimage.io.imread(self.scan_path, as_gray=True)

    @property
    def compacted_text(self)-> str:
        """Return self.text with new lines and blank spaces removed.

        Returns:
            str: self.text with '\n's and ' 's removed.
        """
        with open(self.transcription_path) as f:
            lines = f.readlines()
        text = "".join(lines)
        text = text.replace(" ", "").replace("\n", "")
        return text

    def store_image(self, image: np.ndarray, name: str, check_contrast: bool = False):
        """Store image in self.directory as a file.

            self.binary_extracts determines if the image is stored as a
            binary image or if its labels are preserved.

        Args:
            image (np.ndarray): The image as an ndarray
            name (str): The name for the storing file, its extension
                will be self.image_ext.
            check_contrast (bool, optional): Display contrast warnings.
                Defaults to False.
        """
        if self.binary_extracts:
            self.store_image_binarily(image,name,check_contrast)
        else:
            self.store_image_with_labels(image,name,check_contrast)

    def retrieve_image(self, name: str):
        """Retrieve stored image from its name.

            self.binary_extracts determines if the image is considered as a
            binary image or a labeled image. Hence, the DataManager that
            stores the image and the one that retrieves the image must
            have the same binary_extracts attribute.

        Args:
            name (str): The name used when the image was stored.

        Returns:
            np.ndarray: The image in its original format.
        """
        if self.binary_extracts:
            return self.retrieve_binary_image(name)
        else:
            return self.retrieve_image_with_labels(name)



    def store_image_with_labels(
        self, image: np.ndarray, name: str, check_contrast: bool = False
    ):
        """Store labeled image in self.directory as a file.

        Args:
            image (np.ndarray): The image as an ndarray
            name (str): The name for the storing file, its extension
                will be self.image_ext.
            check_contrast (bool, optional): Display contrast warnings.
                Defaults to False.
        """
        if "." in name:
            raise ValueError(
                "No dot is expected in name, the file "
                + "format is specified and added automatically by the "
                + "store_image method."
            )
        name += self.image_ext
        path_to_image = os.path.join(self.directory, name)

        skimage.io.imsave(
            path_to_image,
            2**16 - 1 - self.contrast_factor * image,
            check_contrast=False,
        )

    def retrieve_image_with_labels(self, name: str) -> np.ndarray:
        """Retrieve stored labeled image from its name.

        Args:
            name (str): The name used when the image was stored.

        Returns:
            np.ndarray: The image in its original format.
        """
        if "." in name:
            raise ValueError(
                "No dot is expected in name, the file "
                + "format is specified and added automatically by the "
                + "retrieve_image method."
            )
        name += self.image_ext
        path_to_image = os.path.join(self.directory, name)
        image = skimage.io.imread(path_to_image, as_gray=True)
        return (2**16 - 1 - image) // self.contrast_factor

    def store_image_binarily(
        self, image: np.ndarray, name: str, check_contrast: bool = False
    ):
        """Convert image to a binary one and store it in self.directory.

        Args:
            image (np.ndarray): The image as an ndarray
            name (str): The name for the storing file, its extension
                will be self.image_ext.
            check_contrast (bool, optional): Display contrast warnings.
                Defaults to False.
        """
        if "." in name:
            raise ValueError(
                "No dot is expected in name, the file "
                + "format is specified and added automatically by the "
                + "store_image method."
            )
        name += self.image_ext
        path_to_image = os.path.join(self.directory, name)

        image=255*(~(image>0)).astype('uint8')

        skimage.io.imsave(
            path_to_image,
            image,
            check_contrast=False,
        )

    def retrieve_binary_image(self, name: str) -> np.ndarray:
        """Retrieve stored binary image from its name.

        Args:
            name (str): The name used when the image was stored.

        Returns:
            np.ndarray: The image in its original format.
        """
        if "." in name:
            raise ValueError(
                "No dot is expected in name, the file "
                + "format is specified and added automatically by the "
                + "retrieve_image method."
            )
        name += self.image_ext
        path_to_image = os.path.join(self.directory, name)
        image = skimage.io.imread(path_to_image, as_gray=True)
        return ~(image>0)

    def store_glyph(
        self, matched_glyph: MatchedGlyph, name: str, with_satellites=True
    ) -> None:
        """Store the matched_glyph on disk."""

        # We could wish to normalize images for further comparison
        # but we postpone this issue
        # normalized_glyph = matched_glyph.glyph.normalized_image()
        glyph_image = matched_glyph.glyph.image
        if not with_satellites:
            glyph_image = (glyph_image == 1).astype("uint16")
        sy, sx = parser.bounding_box_slices(glyph_image)
        normalized_glyph = glyph_image[:, sx]

        self.store_image(normalized_glyph, name)

        path_to_csv = os.path.join(self.directory, "glyphs_metadata.csv")

        with open(path_to_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, dialect="excel")
            writer.writerow(
                [
                    name,
                    matched_glyph.string,
                    matched_glyph.glyph.char_height,
                    matched_glyph.glyph.char_width,
                    matched_glyph.glyph.char_thickness,
                    matched_glyph.glyph.line_height,
                ]
            )

    # The output of the next function will be of the format prescribed below.
    GlyphMetadata = namedtuple(
        "GlyphMetadata",
        [
            "name",
            "string",
            "char_height",
            "char_width",
            "char_thickness",
            "line_height",
        ],
    )

    def retrieve_glyph_metadata(self, name: str) -> GlyphMetadata:
        """Retrieve MatchedGlyph metada from name.

        Args:
            name (str): name of the stored MatchedGlyph

        Returns:
            GlyphMetadata: The named tuple with the required entries.
        """
        path_to_csv = os.path.join(self.directory, "glyphs_metadata.csv")
        with open(path_to_csv, newline="") as csvfile:
            reader = csv.reader(csvfile, dialect="excel")
            for row in reader:
                if name == row[0]:
                    fixed_row = row[:2] + list(map(int, row[2:]))
                    return self.GlyphMetadata._make(fixed_row)
        raise ValueError(
            "We could not find an entry with this name in " + "the metadata."
        )

    def store_cut_info(self, cut_info: CutInfo, name: str) -> None:
        """Store the CutInfo on disk."""

        # We could wish to normalize images for further comparison
        # but we postpone this issue
        # normalized_subword = cut_info.subword.normalized_image()
        normalized_subword = cut_info.subword.image

        self.store_image(normalized_subword, name)

        path_to_csv = os.path.join(self.directory, "cut_info_metadata.csv")

        with open(path_to_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, dialect="excel")
            writer.writerow(
                [
                    name,
                    cut_info.quality,
                    cut_info.cutting_shape_chars[0],
                    cut_info.cutting_shape_chars[1],
                    cut_info.cutting_shape_chars[2],
                    cut_info.subword.char_height,
                    cut_info.subword.char_width,
                    cut_info.subword.char_thickness,
                    cut_info.subword.line_height,
                ]
            )

    # The output of the next function will be of the format prescribed below.
    CutInfoMetadata = namedtuple(
        "CutInfoMetadata",
        [
            "name",
            "quality",
            "cutting_shape_chars",
            "char_height",
            "char_width",
            "char_thickness",
            "line_height",
        ],
    )

    def retrieve_cut_info_metadata(self, name: str) -> CutInfoMetadata:
        """Retrieve CutInfo metada from name.

        Args:
            name (str): name of the stored CutInfo

        Returns:
            CutInfoMetadata: The named tuple with the required entries.
        """
        path_to_csv = os.path.join(self.directory, "cut_info_metadata.csv")
        with open(path_to_csv, newline="") as csvfile:
            reader = csv.reader(csvfile, dialect="excel")
            for row in reader:
                if name == row[0]:
                    fixed_row = (
                        row[:1]
                        + [row[1] == "True"]
                        + [tuple(map(int, row[2:5]))]
                        + list(map(int, row[5:]))
                    )
                    return self.CutInfoMetadata._make(fixed_row)
        raise ValueError(
            "We could not find an entry with this name in " + "the metadata."
        )

    def set_data_counters(
        self, matched_glyph_counter: int = 0, cut_info_counter: int = 0
    ) -> None:
        """Set the counters for stored MatchedGlyphs and CutInfos.

        Args:
            matched_glyph_counter (int, optional): The new value for the
                MatchedGlyphs counter. Defaults to 0.
            cut_info_counter (int, optional): The new value for the
                CutInfos counter. Defaults to 0.
        """
        self.matched_glyph_counter = matched_glyph_counter
        self.cut_info_counter = cut_info_counter

    def store_data(
        self, matched_glyphs: list[MatchedGlyph], cut_infos: list[CutInfo]
    ) -> None:
        """Store the passed lists of objects?

        Args:
            matched_glyphs (list[MatchedGlyph]): A list of MatchedGlyphs
                to store.
            cut_infos (list[CutInfo]): A list of CutInfos to store.
        """
        # Storing the MatchedGlyphs.
        for glyph in matched_glyphs:
            name = 'char_' + str(self.matched_glyph_counter)
            self.store_glyph(glyph, name=name)
            self.matched_glyph_counter += 1
        # Storing the CutInfos.
        for cut_info in cut_infos:
            name = "cut_" + str(self.cut_info_counter)
            self.store_cut_info(cut_info, name=name)
            self.cut_info_counter += 1
