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
    The parser module
-------------------------

Gather facilities for manipulation of handwritten text images:
extraction of lines, words, characters, etc.

"""

from __future__ import annotations
from typing import Any
import string as stringlib
import warnings
import os
import sys

import skimage
import numpy as np
import scipy.ndimage as ndimage
from numba import jit

from . import log_config

logger = log_config.logging.getLogger("hwtools.parser")


def _front_pixel(binary_image):
    support = np.where(binary_image)
    if support[0].size:
        i = support[0][0]
        j = support[1][0]
        return (i, j)
    else:
        logger.warning("The binary image has no front pixel, returning None.")
        return None


def _sanitized_labels(labels):
    # This function is oriented towards 'uint*' types.
    # It should work in many other cases but care should be taken
    # in case of positively and negatively valued labels arrays,
    # regarding the type of the returned matrix.
    sanitized = np.zeros(labels.shape, dtype=labels.dtype)
    vals = list(np.unique(labels))
    vals.pop(0)

    new_val = 1
    for val in vals:
        sanitized[labels == val] = new_val
        new_val += 1
    return sanitized


def horizontal(
    img: np.ndarray,
    centers: list[int],
    thickness: int,
    as_list: bool = False,
) -> np.ndarray() | list[np.ndarray]:
    """Return horizontal lines with prescribed thickness and centers.

    Args:
        img (np.ndarray): The image on which the lines should be
            overlayed.
        centers (list[int]): the heights at which the lines must appear.
        thickness (int): the prescribed thickness for the lines.
        as_list (bool, optional): If the lines are beeing returned in a
            single image. Defaults to False.

    Returns:
        np.ndarray() | list[np.ndarray]: A boolean image that contains
        all the horizontal lines if as_list==False, else a list of
        such images, each containing a unique line.

    """
    lines_list = []
    for i in centers:
        line = np.zeros(img.shape, dtype="bool")
        line[max(0, i - thickness // 2) : i + thickness // 2] = True
        lines_list.append(line)
    if as_list:
        return lines_list
    else:
        return sum(lines_list).astype("bool")


def bounding_box_slices(image: np.ndarray) -> tuple[slice, slice]:
    """Return the slices that define the bounding box of image.

    Args:
        image (np.ndarray): a 2d ndarray

    Raises:
        ValueError: "The input image is not 2-d"
        ValueError: "The given array is null, its bounding
            box is not well defined."

    Returns:
        tuple[slice,slice]: (s_y,s_x) the pair of slices, s_y for
            the vertical axes, s_x for the horizontal one.
    """
    if image.ndim != 2:
        raise ValueError("The input image is not 2-d")
    if np.all(image == 0):
        raise ValueError(
            "The given array is null, its bounding box is not well defined."
        )
    support = np.where(image)
    x_min = np.amin(support[1])
    x_max = np.amax(support[1])
    y_min = np.amin(support[0])
    y_max = np.amax(support[0])
    return (slice(y_min, y_max), slice(x_min, x_max))


# To optimize: eliminate extraneous version of bounding box, another version is
# appearing as a method below.
@jit(nopython=True)
def bounding_box(image: np.ndarray) -> np.ndarray:
    """Return the bounding box of a 2d image.

    Args:
        image (np.ndarray): A np.ndarray with image.ndim==2.
            It represents a grayscale or binary image.

    Returns:
        np.ndarray: The smallest rectangular subimage that contains all
            non-zero pixels.
    """

    sy, sx = bounding_box_slices(image != 0)
    return image[sy, sx]


# we could try to optimize compsize below stopping once
#  an expected threshold is reached
@jit(nopython=True)
def comp_size(labels: np.ndarray, val: int) -> int:
    """Return the size of a component in a labeled image.

    The size is understood as the number of pixels.

    Args:
        labels (np.ndarray): an integer valued 2d ndarray
        val (int): the value whose pixels we want to count.

    Returns:
        int: The number of occurences of value among the pixels of
            labels.
    """
    return np.sum((labels == val).astype("uint16"))


@jit(nopython=True)
def hand_made_binary_dilation(
    img: np.ndarray, footprint: np.ndarray
) -> np.ndarray:
    """Naive binary dilation, cheaper than skimage's for sparse images.

    Args:
        img (np.ndarray): The binary image to which the dilation must be
            applied
        footprint (np.ndarray): The footprint for the dilation algorithm.

    Returns:
        np.ndarray: a binary image obtained by binary dilation of img wrt
            footprint.
    """
    h, w = img.shape
    a, b = footprint.shape
    u = a // 2
    v = b // 2
    R = np.zeros(img.shape, dtype="bool")
    support = np.where(img)

    for i in range(len(support[0])):
        y = support[0][i]
        x = support[1][i]
        # We describe the relevant bounds for the window in img centered
        # at (y,x).
        top = max(0, y - u)
        bottom = min(y + u + 1, h)
        left = max(x - v, 0)
        right = min(x + v + 1, w)
        # C = adapted cropped footprint
        C = footprint[
            u - (y - top) : u + bottom - y, v - (x - left) : v + right - x
        ]
        A = R[top:bottom, left:right]
        R[top:bottom, left:right] = A | C
    return R


class PageExtract:
    """A class for page extracts, to retain context information.

    It has strong relations with the class Page.
    """

    def __init__(
        self,
        image: np.ndarray,
        char_height: int,
        char_thickness: int,
        char_width: int,
        corner_y: int,
        corner_x: int,
        parent_page_shape: tuple[int, int],
    ):
        """Instantiate self.

        Args:
            image (np.ndarray): The image of the page extract
            char_height (int): An estimate of the characters' height.
            char_thickness (int): An estimate of the characters'
                thickness.
            char_width (int): An estimate of the characters' width.
            corner_y (int): the top-left corner y coordinate within the
                parent page.
            corner_x (int): the top-left corner x coordinate within the
                parent page.
            parent_page_shape (tuple[int,int]): The shape of the parent
            page.
        """
        self.image = image
        self.corner_y = corner_y
        self.corner_x = corner_x
        self.parent_page_shape = parent_page_shape
        self.char_height = char_height
        self.char_thickness = char_thickness
        self.char_width = char_width

    def in_page(self) -> Any:
        """Put the page extract in its original page.

        Returns:
            PageExtract: a page extract of the whole ambient page dimensions,
            with the input PageExtract at its position.
        """
        page = np.zeros(self.parent_page_shape)
        page[
            self.corner_y : self.corner_y + self.image.shape[0],
            self.corner_x : self.corner_x + self.image.shape[1],
        ] = self.image

        return PageExtract(
            page,
            self.char_height,
            self.char_thickness,
            self.char_width,
            0,
            0,
            self.parent_page_shape,
        )

    def __getitem__(
        self, positions: tuple[slice | int, slice | int]
    ) -> PageExtract:
        """Mimick behaviour of __getitem__ for np.ndarray.


        Args:
            positions (tuple[slice | int, slice | int]): tuple of
            indices or slices.

        Returns:
            PageExtract: The page extract corresponding to the sliced
                self.image seen within parent_page.
        """

        def int2slice(x: int | slice) -> slice:
            """Convert integer index or slice to the corresponding slice.

            Args:
                x (int | slice): input integer or slice. Beware that
                the step in every slice is implicitely considered as 1.

            Returns:
                slice: corresponding length 1 slice.
            """

            if isinstance(x, int):
                return slice(x, x + 1)
            else:
                if x.start is None:
                    start = 0
                else:
                    start = x.start
                return slice(start, x.stop)

        if not isinstance(positions, tuple):
            positions = (positions,)
        if all(isinstance(elem, slice | int) for elem in positions):
            image = self.image[positions]
            slices = list(int2slice(i) for i in positions)
        else:
            logger.warning(
                "The given tuple of slices/indices is not valid, returning None."
            )
            return None
        corner_y = self.corner_y + slices[0].start
        corner_x = self.corner_x + slices[1].start
        return PageExtract(
            image,
            self.char_height,
            self.char_thickness,
            self.char_width,
            corner_y,
            corner_x,
            self.parent_page_shape,
        )

    def bounding_box(self) -> np.ndarray:
        """Return the bounding box of self.

        Returns:
            np.ndarray: The smallest rectangular subimage that contains
                all non-zero pixels of self.
        """
        sy, sx = bounding_box_slices(self.image)
        return self[sy, sx]

    def changed_image(self, new_image: np.ndarray) -> PageExtract:
        """Return copy of self with image attribute changed to new_image.

        Args:
            new_image (np.ndarray): The new image

        Returns:
            PageExtract: Copy of self with image attribute changed to
                new_image.
        """
        return PageExtract(
            new_image,
            self.char_height,
            self.char_thickness,
            self.char_width,
            self.corner_y,
            self.corner_x,
            self.parent_page_shape,
        )

    # def union(self, other):
    #     if self.parent_page_shape != other.parent_page_shape:
    #         warnings.warn(
    #             "The two page extracts do not have the same "
    #             + "parent_page_shape. Make sure these are the right objects."
    #         )
    #     corner_y = min(self.corner_y, other.corner_y)
    #     corner_x = min(self.corner_x, other.corner_x)


class Line:
    """The class for the lines of the treated scanned  documents."""

    def __init__(self, page_extract: PageExtract, line_height: int):
        """Instantiate Line objects.

        Args:
            page_extract (PageExtract): The underlying PageExtract
            line_height (int): the line height within
                page_extract.parent_page.
        """

        self.page_extract = page_extract
        self.line_height = line_height
        self._words = None

    @property
    def image(self):
        """Return self.page_extract.image."""

        return self.page_extract.image

    @property
    def complete_line(self):
        """Return self.page_extract.image != 0."""

        return self.page_extract.image != 0

    @property
    def basic_line(self):
        """Return self.page_extract.image == 1."""

        return self.page_extract.image == 1

    @property
    def satellites(self, as_list=False) -> list[np.ndarray] | np.ndarray:
        """Return the 'satellite' components in self.image.

        Args:
            as_list (bool, optional): Whether to return the satellites
                together as a unique binary image or as a list of
                distinct connected ones. Defaults to False.

        Returns:
            _list[np.ndarray] |np.ndarray: the satellites as a (list of)
                binary images.
        """
        if as_list:
            return [
                self.page_extract.image == i
                for i in np.unique(self.complete_line)
                if i > 1
            ]
        return self.page_extract.image > 1

    @property
    def char_thickness(self):
        """Return self.page_extract.char_thickness."""
        return self.page_extract.char_thickness

    @property
    def char_height(self):
        """Return self.page_extract.char_height."""
        return self.page_extract.char_height

    @property
    def char_width(self):
        """Return self.page_extract.char_width."""
        return self.page_extract.char_width

    def words(
        self,
        min_width: int | None = None,
        max_gap: int | None = None,
        margin: int | None = None,
        line_thickness: int | None = None,
    ) -> list[Words]:
        """Return the Words extracted from the line.

        If a parameter is received as None, it will be given a default
        integer value, in function of self.
        Afterwards, the algorithm is as follows: a thickline at
        line_height is drawn 'over' line.image,
        with thickness determined by line_thickness. Every connected
        component of line that meets this thick line is considered as
        participating to the basic_line of the line. (The other
        components are termed satellites.) Then, if  the connected
        components are grouped as the equivalence relation obtained by
        saturation of the relation
        'the horizontal projections are far from at most max_gap'.

        Then, the groups with horizontal projection width less than
        min_width are discarded, considered as noise.

        For each of the remaining groups, we consider the smallest interval
        that contains its horizontal projection, enlarge it on both side
        by margin. The front pixels that lie over these enlarged intervals
        define the various Words (one per interval).


        Args:
            min_width (int | None, optional): _description_. Defaults to None.
            max_gap (int | None, optional): _description_. Defaults to None.
            margin (int | None, optional): _description_. Defaults to None.
            line_thickness (int | None, optional): _description_. Defaults to None.

        Returns:
            list[Words]: The calculated Words from self.
        """
        if (
            self._words is None
            or min_width is not None
            or max_gap is not None
            or margin is not None
            or line_thickness is not None
        ):
            self._words = self._find_words(
                min_width=min_width,
                max_gap=max_gap,
                margin=margin,
                line_thickness=line_thickness,
            )
        return self._words

    def _find_words(
        self, min_width=None, max_gap=None, margin=None, line_thickness=None
    ):
        if min_width is None:
            min_width = self.char_thickness
        if max_gap is None:
            max_gap = self.char_width // 2
        if margin is None:
            margin = self.char_width // 4
        if line_thickness is None:
            line_thickness = self.char_height

        line_height = self.line_height
        thickline = horizontal(self.basic_line, [line_height], line_thickness)
        support = np.unique(np.where(thickline & self.complete_line)[1])

        # Another options would be
        # np.unique(np.where(self.basic_line)[1])
        # but this raises the risk of connection of two words by upper symbols.

        support_bounds = []
        # Support_bounds contains the frontier points of the support,
        # in increasing order.
        for i in support:
            if i - 1 not in support:
                support_bounds.append(i)
            if i + 1 not in support:
                support_bounds.append(i)

        # We perform some curation of support_bounds,
        # to ignore small gaps in a word.
        i = 1
        while i < len(support_bounds) - 1:
            if support_bounds[i + 1] - support_bounds[i] < max_gap:
                support_bounds = support_bounds[:i] + support_bounds[i + 2 :]
            else:
                i += 2

        # Afterwards we perform a similar treatment to avoid empty words,
        # coming from noise.
        i = 0
        while i < len(support_bounds) - 1:
            if support_bounds[i + 1] - support_bounds[i] < min_width:
                support_bounds = support_bounds[:i] + support_bounds[i + 2 :]
            else:
                i += 2
        return [
            Word(
                self.page_extract[
                    :,
                    max(0, support_bounds[2 * i] - margin) : min(
                        1 + support_bounds[2 * i + 1] + margin,
                        self.page_extract.image.shape[1],
                    ),
                ],
                self.line_height,
            )
            for i in range(len(support_bounds) // 2)
        ]


class Word(Line):
    """A subclass of Line that deals with images of handwrittent words."""

    def __init__(self, page_extract, line_height):
        super().__init__(page_extract, line_height)
        self._cutting_shapes = None

    # We override the words method from superclass Line.
    def words(
        self, min_width=None, max_gap=None, margin=None, line_thickness=None
    ):
        """Return self and print a warning."""
        warnings.warn(
            'The "words" method is being called '
            + "for an instance of the Word subclass."
            + " This is not very relevant:"
            + "in this subclass the method reduces to the identity map."
        )
        return [self]

    def word_part(self, sub_image: np.ndarray) -> Word:
        """Return a copy of self with  self.image replaced by subimage.

        Args:
            sub_image (np.ndarray): the replacement image for self.image.
        """
        page_extract = self.page_extract.changed_image(sub_image)
        return Word(page_extract, self.line_height)

    def cut(self, shapes: list[np.ndarray]) -> Word:
        """Return a copy of self, but cut, removing the pixels in shapes.

        Args:
            shapes (list[np.ndarray]): regions of self.image to turn
                white. (cutting shapes)

        Returns:
            Word: Cut version of self.
        """
        sub_image = self.image.copy()
        for shape in shapes:
            sub_image[shape] = 0
        return self.word_part(sub_image)

    # We use a similar strategy as in find_words to get the components
    # of the word associated with the connected components of its basic line.
    # We observe that for every satellite to be attributed to some basic
    # component, the sat_margin parameter used here should be bigger than
    # max_gap used in _find_words
    def find_word_components(
        self,
        sat_margin: int | None = None,
        line_thickness: int | None = None,
        # consider_punctuation: bool = False,
        # punctuation_margin: int | None = None,
    ) -> list[Word]:
        """Find the components of a Word.

        These components are calculated as the connected components of
        self.basic_line (as defined for find_words, in terms of
        line_thickness) plus some satellites of Word, that are considered
        as belonging to the component if any pixel of them lie over the
        sat_margin enlarged projection of the component.

        Args:
            sat_margin (int | None, optional): The tolerance for
                satellites attribution. Defaults to None. If None is
                passed a value is fixed automatically.
            line_thickness (int | None, optional): The thickness that
                determines the basic_line computation. Defaults to None.
                If None is passed a value is fixed automatically.


        Returns:
            list[Word]: The list of components of self, as subWords.
        """
        if sat_margin is None:
            sat_margin = 0

        # if punctuation_margin is None:
        #     punctuation_margin = self.char_width // 3
        if line_thickness is None:
            line_thickness = self.char_height // 2
        # TO OPTIMIZE: we are not using the previous labelling of satellites by
        # the words method of the Line class.
        local_line = self.complete_line
        line_height = self.line_height
        thickline = horizontal(local_line, [line_height], line_thickness)
        labels = skimage.measure.label(local_line)
        basic_comp_projections = []
        satellites_projections = []

        for i in range(1, np.amax(labels) + 1):
            comp = labels == i

            # For satellites attribution,
            # we will compare vertical projections of a given sat and
            # the various basic_comps.
            # We represent the projections of these connected subspaces
            # as intervals.
            comp_supp = np.unique(np.where(labels == i)[1])
            a, b = np.amin(comp_supp), np.amax(comp_supp)

            # Also sort the components in two categories: satellites
            # and basic line comps.
            if np.any(comp & thickline):
                basic_comp_projections.append(((a, b), i))
            else:
                satellites_projections.append(((a, b), i))

        complete_components = []
        for pair, i in basic_comp_projections:
            complete_component = (labels == i).astype("uint16")
            a, b = pair
            A = a - sat_margin
            B = b + sat_margin

            for couple, j in satellites_projections:
                u, v = couple
                if v > A and u < B:
                    complete_component[labels == j] = j + 1
            complete_component = _sanitized_labels(complete_component)
            complete_components.append(self.word_part(complete_component))

        # # This branch is to include punctuation as autonomous word components if necessary.
        # # They could otherwise appear only as satellites of other components.
        # if consider_punctuation:
        #     basic_comp_intervals = [
        #         projection[0] for projection in basic_comp_projections
        #     ]
        #     for couple, j in satellites_projections:
        #         augmented_list = basic_comp_intervals + [couple]
        #         augmented_list.sort(key=lambda t: t[0])
        #         i = augmented_list.index(couple)
        #         if len(augmented_list) > i + 1:
        #             top_ok = (
        #                 augmented_list[i + 1][0] + punctuation_margin
        #             ) > couple[1]
        #         else:
        #             top_ok = True
        #         if i > 0:
        #             bottom_ok = (
        #                 augmented_list[i - 1][1] - punctuation_margin
        #             ) < couple[0]
        #         else:
        #             bottom_ok = True
        #         if bottom_ok and top_ok:
        #             # We mark detected punctuation components with the
        #             # label '2', to distinguish them from standard complete
        #             # components that have '1' entries, corresponding to their
        #             # basic components.
        #             # Remember that punctuation symbols may contribute several
        #             # of these punctuation components, because these are
        #             # connected.
        #             complete_component = 2 * (labels == j).astype("uint16")
        #             complete_components.append(
        #                 self.word_part(complete_component)
        #             )

        # We will use the next function to sort components from
        # left to right.
        def mean_x(component):
            a = np.where(component > 0)[1]
            size = a.shape[0]
            if size == 0:
                raise ValueError(
                    "mean_x received a component with no positive entry."
                )
            else:
                return np.sum(a) / size

        complete_components.sort(
            key=lambda component: mean_x(component.complete_line)
        )

        return complete_components

    @property
    def cutting_shape_thickness(self) -> int:
        """Prescribe the thickness of cutting shapes for word cutting.

        Returns:
            int: (3 * self.char_thickness) // 2
        """
        return (3 * self.char_thickness) // 2

    def _is_critical(self, point, thickness=None, window_radius=None):
        if thickness is None:
            thickness = self.cutting_shape_thickness
        if window_radius is None:
            window_radius = char_width

        y, x = point
        cut_word = self.basic_line.copy()
        cut_word[y - thickness // 2 : y + thickness // 2, x] = False

        # The ideal solution would be the one below
        # return np.amax(skimage.measure.label(cut_word))
        #        > np.amax(skimage.measure.label(word))
        # however we take into account computational costs
        # and consider shortened words.
        shortened_word = self.basic_line[
            :, max(0, x - self.char_width) : x + self.char_width
        ]
        shortened_cut_word = cut_word[
            :, max(0, x - self.char_width) : x + self.char_width
        ]
        labels = skimage.measure.label(shortened_cut_word)
        disconnects = np.amax(labels) > np.amax(
            skimage.measure.label(shortened_word)
        )

        if disconnects:
            # We check the local components are either
            # on the left or on the right of the point
            r = window_radius
            window = cut_word[y - r : y + r, x]
            return not np.any(window)
        else:
            return False

    def _find_thin_places(self, thickness=None):
        if thickness is None:
            thickness = self.cutting_shape_thickness

        thin_places = []
        for x in range(self.basic_line.shape[1]):
            s = self.basic_line[:, x]

            support = np.where(s > 0)[0]
            support_bounds = []
            # support_bounds contains the frontier points of the support,
            # in increasing order.
            for i in support:
                if i - 1 not in support:
                    support_bounds.append(i)
                if i + 1 not in support:
                    support_bounds.append(i)

            for i in range(len(support_bounds) // 2):
                if (
                    support_bounds[2 * i + 1] - support_bounds[2 * i]
                    <= 2 * thickness // 2
                ):
                    thin_places.append(
                        (
                            (support_bounds[2 * i + 1] + support_bounds[2 * i])
                            // 2
                            + 1,
                            x,
                        )
                    )

        return thin_places

    # Below, we check at once if a place is of minimal height in the
    # vertical slice of the word, and if the word is easily cut at this place
    # (partially checking condition 1 above, and totally checking condition 2)
    # this is a better version of _find_thin_places
    def _find_low_and_thin_places(self, thickness=None):
        if thickness is None:
            thickness = self.cutting_shape_thickness

        low_and_thin_places = []
        for x in range(self.basic_line.shape[1]):
            s = self.basic_line[:, x]

            support = np.where(s > 0)[0]
            support_bounds = []
            # Support_bounds contains the frontier points of the support,
            #  in increasing order.
            for i in support:
                if i - 1 not in support:
                    support_bounds.append(i)
                if i + 1 not in support:
                    support_bounds.append(i)

            if (
                support_bounds
                and support_bounds[-1] - support_bounds[-2]
                <= 2 * thickness // 2
            ):
                low_and_thin_places.append(
                    ((support_bounds[-1] + support_bounds[-2]) // 2 + 1, x)
                )

        return low_and_thin_places

    def _is_not_too_excentric(
        self, p, line_height, tolerated_excentricity=None
    ):
        if tolerated_excentricity is None:
            tolerated_excentricity = self.char_height // 2
        y, x = p
        return (
            y > line_height - tolerated_excentricity
            and y < line_height + tolerated_excentricity
        )

    def _find_candidates(self, thickness=None, window_radius=None):
        if thickness is None:
            thickness = self.cutting_shape_thickness
        if window_radius is None:
            window_radius = self.char_height // 2
        low_and_thin_places = self._find_low_and_thin_places(
            thickness=thickness
        )

        candidates = np.zeros(self.basic_line.shape, dtype="bool")
        line_height = self.line_height
        for p in low_and_thin_places:
            if self._is_not_too_excentric(
                p, line_height
            ) and self._is_critical(
                p, thickness=thickness, window_radius=window_radius
            ):
                candidates[p] = True
        return candidates

    # To optimize : the candidates above are given as front pixels in an
    # image and then manipulated in coordinates below, choose one representation
    # and stick to it: coordinates seem to be a good choice.
    @staticmethod
    def _find_lowest_points(candidates):
        points = []
        projection = np.unique(np.where(candidates)[1])

        for x in projection:
            y = np.where(candidates[:, x])[0][-1]
            if x - 1 not in projection:
                current_min = y
                x_min = x
            else:
                # In the used coordinates y > current_min  means a lower point.
                if y > current_min:
                    current_min = y
                    x_min = x
            if x + 1 not in projection:
                points.append((current_min, x_min))
        points.sort(key=lambda t: t[1])
        return points

    @staticmethod
    def _touches(
        p: tuple[int, int], comp: np.ndarray, side: str | None = None
    ) -> bool:
        """Decide if p touches comp on the right, the left or any side.

        Args:
            p (tuple[int, int]): A point in an image, coords (y,x)
            comp (np.ndarray): A set of pixels in this image,
                given as a boolean array.
            side (str | None, optional): The side to consider: None,
                'right' or 'left'.
                Defaults to None. If None is passed, any side is
                considered valid.

        Returns:
            bool: True if comp is on the side-hand of p,
                False otherwise. If side is None, True if and only if
                p touches comp.
        """

        if isinstance(side, str):
            side = side.lower()

        if side is None:
            dilated_comp = hand_made_binary_dilation(
                comp, np.ones((3, 3), dtype="bool")
            )
        elif side == "right":
            dilated_comp = hand_made_binary_dilation(
                comp, np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype="bool")
            )
        elif side == "left":
            dilated_comp = hand_made_binary_dilation(
                comp, np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype="bool")
            )
        else:
            raise ValueError(
                "The side argument was not passed "
                + 'correctly. It must be either "right", "left" or None.'
            )
        return bool(dilated_comp[p])

    @staticmethod
    def _inbetween_piece(
        p: tuple[int, int],
        q: tuple[int, int],
        cut_comp: np.ndarray,
        side_sensitive=False,
    ) -> np.ndarray | None:
        """Return the piece of cut_comp which lies between p and q.


        Args:
            p (tuple[int, int]): The left-hand point, in the form (y,x).
            q (tuple[int, int]): The right-hand point, in the form
                (y0,x0). We suppose x0>x.
            cut_comp (np.ndarray): A binary image.
            side_sensitive (bool): whether one takes into account the
                side on which p and q are touched by the sought component.

        Returns:
            np.ndarray | None: if side_sensitive is True,
                 a connected component of cut_comp that
                touches p on the right-hand side of p and q on the
                left-hand of q; provided it exists. In this case, it is
                given as a binary image. If such a component does not
                exist, None is returned.


                If side_sensitive is False p or q may touch the
                sought component on any side. If such a component does
                not exist, None is returned.
        """
        if p[1] >= q[1]:
            raise ValueError("The point p does not lie on the left of q.")
        labels = skimage.measure.label(cut_comp != 0)
        p_criterion = "right" if side_sensitive else None
        q_criterion = "left" if side_sensitive else None

        ok_with_p = []
        for i in range(1, np.amax(labels) + 1):
            if Word._touches(p, labels == i, p_criterion):
                ok_with_p.append(labels == i)
        if not ok_with_p:
            return None
        for chunk in ok_with_p:
            if Word._touches(q, chunk, q_criterion):
                return chunk

    def _event_detected(
        self,
        p: tuple[int, int],
        q: tuple[int, int],
        height_tolerance: int,
        pixel_tolerance: int,
        thickness: int,
    ):
        a = p[1]
        b = q[1]
        if a > b:
            logger.warning(
                "p[1] > q[1]: (p,q)="
                + str((p, q))
                + "We return the result considering these points as permuted."
            )
            return _event_detected(q, p, component, tolerance=tolerance)

        if a + 1 == b or a == b:
            return False

        p_shape = self._cutting_shape_from_point(p, thickness)
        q_shape = self._cutting_shape_from_point(q, thickness)
        cut_comp = self.basic_line
        cut_comp[p_shape | q_shape] = 0
        piece = Word._inbetween_piece(p, q, cut_comp)
        if piece is None:
            return False

        max_heights = []
        min_heights = []

        for x in range(a + 1, b):
            slice_support = np.where(piece[:, x])[0]
            max_heights.append(piece.shape[0] - np.amin(slice_support))
            min_heights.append(piece.shape[0] - np.amax(slice_support))
        max_heights = np.array(max_heights)
        min_heights = np.array(min_heights)
        pixel_number = np.sum(piece[:, a + 1 : b].astype("uint8"), axis=0)

        if (
            np.amax(max_heights) > max_heights[0] + height_tolerance
            or np.amin(min_heights) < min_heights[0] - height_tolerance
        ):
            # height event
            logger.info("height event")
            return True
        elif np.amin(pixel_number) < np.amax(pixel_number) - pixel_tolerance:
            # pixel event
            logger.info("pixel_event")
            logger.info(
                str((np.amin(pixel_number), np.amax(pixel_number), p, q))
            )
            return True
        else:
            return False

    # to optimize: the two methods below were added lately, they could be used
    # to refactor part of the former code.
    def _cutting_shape_from_point(self, p, thickness=None):
        if thickness is None:
            thickness = self.cutting_shape_thickness
        y, x = p
        shape = np.zeros(self.complete_line.shape, dtype="bool")
        shape[y - thickness // 2 : y + thickness // 2, x] = True
        return shape

    @staticmethod
    def _cutting_point_from_shape(shape):
        support = np.where(shape)
        x = int(np.median(support[1]))
        y = int(np.median(support[0]))

        return (y, x)

    def _creates_too_small_comp(
        self,
        shape: np.ndarray,
        too_small_comp_radius: int,
        min_size: int,
    ) -> bool:
        """Check if cutting at shape creates too small pieces.

        Args:
            shape (np.ndarray): a cutting shape
            window_radius (int): The radius of the window around shape
                center in which we perform the size check.
            min_size (int): the minimal acceptable size for the cut
                components.

        Returns:
            bool: True if a too small component would be created,
                False otherwise.
        """
        logger.debug(
            "Entering _creates_too_small_comp with min_size = "
            + str(min_size)
            + " and too_small_comp_radius = "
            + str(too_small_comp_radius)
        )
        y, x = self._cutting_point_from_shape(shape)
        radius = too_small_comp_radius
        cut_word = self.basic_line.copy()
        cut_word[shape] = False
        shortened_word = self.basic_line[:, max(0, x - radius) : x + radius]
        labels = skimage.measure.label(shortened_word)
        comp_sizes = [
            np.sum((labels == i).astype("uint16"))
            for i in range(1, np.amax(labels) + 1)
        ]
        small_comps_count = [size < min_size for size in comp_sizes].count(
            True
        )

        shortened_cut_word = cut_word[:, max(0, x - radius) : x + radius]
        labels = skimage.measure.label(shortened_cut_word)
        new_comp_sizes = [
            np.sum((labels == i).astype("uint16"))
            for i in range(1, np.amax(labels) + 1)
        ]
        new_small_comps_count = [
            size < min_size for size in new_comp_sizes
        ].count(True)
        logger.debug(
            "small_comps_counts,before cutting: "
            + str(small_comps_count)
            + " after cutting: "
            + str(new_small_comps_count)
        )
        logger.debug("components sizes before cutting: " + str(comp_sizes))
        logger.debug("components sizes after cutting: " + str(new_comp_sizes))

        output = new_small_comps_count > small_comps_count

        logger.debug("returning: " + str(output))

        return output

    # We define _find_letter_cutting_shapes using this _event_detected
    # function.
    # We also take into account the special roles of the beginning and end of
    # the component.
    # Warning: in the auxiliary function below,
    # we are making the assumption that THE BASIC LINE OF SELF IS CONNECTED.

    def _find_letter_cutting_shapes_in_component(
        self,
        height_tolerance,
        pixel_tolerance,
        window_radius,
        thickness,
        too_small_comp_radius,
        min_size,
    ):
        line_height = self.line_height
        logger.info("new_comp")
        shapes = []
        candidates = self._find_candidates(
            thickness, window_radius=window_radius
        )
        lowest_points = Word._find_lowest_points(candidates)
        # We transpose self.basic_line to obtain the (one) leftmost point
        # for free thanks to the output ordering of np.where.
        # We also exploit this to get the rightmost point.
        supp = np.where(self.basic_line.transpose())
        p0 = (supp[1][0], supp[0][0])  # p0 in the form (y,x)
        pinf = (supp[1][-1], supp[0][-1])  # pinf in the form (y,x)

        projection = np.unique(np.where(self.basic_line)[1])
        a = np.amin(projection)
        b = np.amax(projection)

        # We use event_detected to avoid successions of similar cutting shapes.
        filtered_cutting_points = []
        comparison_point = p0
        current_pack = []
        packs = []
        for p in lowest_points:
            if self._event_detected(
                comparison_point,
                p,
                height_tolerance=height_tolerance,
                pixel_tolerance=pixel_tolerance,
                thickness=thickness,
            ):
                packs.append(current_pack)
                current_pack = [p]
                comparison_point = p
            else:
                current_pack.append(p)
        packs.append(current_pack)
        filtered_cutting_points = [
            pack[len(pack) // 2] for pack in packs if pack
        ]

        # We deal here with the special case of useless cutting points close to
        # the end of the component.
        if filtered_cutting_points and not self._event_detected(
            filtered_cutting_points[-1],
            pinf,
            height_tolerance=height_tolerance,
            pixel_tolerance=pixel_tolerance,
            thickness=thickness,
        ):
            filtered_cutting_points.pop(-1)

        for p in filtered_cutting_points:
            shape = self._cutting_shape_from_point(p, thickness)
            shapes.append(shape)

        # We filter once more using _creates_too_small_comp
        for i in range(len(shapes) - 1, -1, -1):
            if self._creates_too_small_comp(
                shapes[i],
                too_small_comp_radius=too_small_comp_radius,
                min_size=min_size,
            ):
                shapes.pop(i)
                filtered_cutting_points.pop(i)

        return (shapes, filtered_cutting_points)

    # Now we use this to propose the cutting shapes for a whole line or word,
    # thus possibly disconnected.
    def _find_letter_cutting_shapes(
        self,
        height_tolerance,
        pixel_tolerance,
        window_radius,
        thickness,
        too_small_comp_radius,
        min_size,
    ):
        line_height = self.line_height
        components = self.find_word_components()
        shapes = []
        points = []
        for component in components:
            (
                temp_shapes,
                temp_points,
            ) = component._find_letter_cutting_shapes_in_component(
                thickness=thickness,
                height_tolerance=height_tolerance,
                pixel_tolerance=pixel_tolerance,
                window_radius=window_radius,
                too_small_comp_radius=too_small_comp_radius,
                min_size=min_size,
            )
            shapes += temp_shapes
            points += temp_points
            indices = list(range(len(shapes)))
            indices.sort(key=lambda i: points[i][1])
            shapes = [shapes[i] for i in indices]
            points.sort(key=lambda point: point[1])
        return (shapes, points)

    def cutting_shapes(
        self,
        height_tolerance: int | None = None,
        pixel_tolerance: int | None = None,
        window_radius: int | None = None,
        thickness: int | None = None,
        too_small_comp_radius: int | None = None,
        min_size: int | None = None,
        recompute: int | None = False,
    ) -> tuple[list[np.ndarray], tuple[int, int]]:
        """Find where to cut self to separate its letters.

        We explain below the aspects of the algorithm that are tunable
        through the function parameters. For this discussion, we suppose
        None of the parameter is passed as None.

        A cut/cutting shape is a 1 pixel width rectangular region in
        self.image. The parameter of the algorithm that fixes the height
        of the sought cutting shapes is 'thickness'. A cutting shape
        should enjoy the property that once it is removed from
        self.basic_component, the latter's number of connected components
        augments.

        We actually check a slightly different property: the shape must
        disconnect the restriction of self.basic_line to a square window
        of radius 'window_radius', we also
        require that no pixel of self.basic_line remains in that window
        on the vertical line that contains the cut.

        For each component of self, as calculated as self.find_components(),
        the algorithm determines a first set of cut candidates that
        respects the conditions above (among others).
        The candidates are ordered from left to right. We add
        Still componentwise, the leftmost and rightmost pixel of
        self.basic_line to the list.
        This list is filtered from left to right, to ensure some kind of
        event happens between two succesive points: either height events
        or pixel events.

         - We consider some height event happens between two
                elements of the list if the part of self.basic_component that links
                them has at least a height variation of 'height_tolerance'.
         - We consider some pixel  event happens between two
                elements of the list if the part of self.basic_component that links
                them has at least a variation of 'pixel_tolerance' for the number
                of pixel in its 1-pixel wide vertical slices.

        Any element of the list is eliminated if no such event happens
        between it and the preceding element (of the updated list).

        [...]

        Finally, We want to remove the cuts that would create too small
        pieces. This is tested in restriction to a square window of radius
        'too_small_comp_radius', the newly created components in that
        window must be bigger or equal than 'min_size' for the cut to be
        acceptable.


        Args:
            height_tolerance (int | None, optional): Tolerance for
                height events. Defaults to None. If None is passed a value
                is fixed automatically.
            pixel_tolerance (int | None, optional): Tolerance for
                pixel events. Defaults to None. If None is passed a value
                is fixed automatically.
            window_radius (int | None, optional): Window size for
                disconnection test. Defaults to None.If None is passed, a
                value is fixed automatically.
            thickness (int | None, optional): The thickness for the
                cutting shapes. Defaults to None. If None is passed a
                value is fixed automatically.
            too_small_comp_radius (int | None, optional): Window size for
                new component size test. Defaults to None. If None is
                passed a value is fixed automatically.
            min_size (int | None, optional): The minimal acceptable size
                in new component size test. Defaults to None. If None is
                passed a value is fixed automatically.
            recompute (int | None, optional): If the cuts should be
                recomputed. Defaults to False. To spare computation,
                it is prefered to store the output after the first call
                of this method. When testing new parameter values, use
                recompute=True.

        Returns:
            tuple[list[np.ndarray],tuple[int,int]]: The list of cutting
            shapes as binary images of the same shape as self.image and
            the list of the centers of these shapes in the coordinates
            of self.image.
        """
        if height_tolerance is None:
            height_tolerance = self.char_height // 4
        if pixel_tolerance is None:
            pixel_tolerance = self.char_thickness
        if window_radius is None:
            window_radius = self.char_height // 6
        if thickness is None:
            thickness = self.cutting_shape_thickness
        if too_small_comp_radius is None:
            too_small_comp_radius = self.char_width
        if min_size is None:
            min_size = (2 * too_small_comp_radius * self.char_thickness) // 3

        if self._cutting_shapes is None or recompute:
            self._cutting_shapes = self._find_letter_cutting_shapes(
                thickness=thickness,
                height_tolerance=height_tolerance,
                pixel_tolerance=pixel_tolerance,
                window_radius=window_radius,
                too_small_comp_radius=too_small_comp_radius,
                min_size=min_size,
            )
        return self._cutting_shapes

    # To DEBUG:
    # The following method needs enhancement, see ValueError call.
    def _normalized_image(
        self, char_height=None, char_thickness=None
    ) -> np.ndarray:
        """Normalize word.


        Normalize word to a given char_height and char_thickness
        and arrange for line_height to be half of the new
        self.image.shape[0].

        Args:
            char_height (int): the sought char_height
            char_thickness (int): the sought char_thickness

        Returns:
            Word: the normalized Word
        """

        if char_height is None:
            char_height = self.char_height
        if char_thickness is None:
            char_thickness = self.char_thickness

        if not (
            char_height == self.char_height
            and char_thickness == self.char_thickness
        ):
            raise ValueError(
                "The normalized_image method does not deal with "
                + "char_height or char_thickness changes for the moment!"
            )

        sy, sx = bounding_box_slices(self.image)
        ymax, xmax = self.image.shape
        y0 = sy.start
        y1 = sy.stop
        h = self.line_height
        r = max(h - y0, y1 - h)
        u0 = h - r
        u1 = h + r
        if u0 < 0:
            new_image = np.vstack(
                [
                    np.zeros((-u0, sx.stop - sx.start), dtype="bool"),
                    self.image[:u1, sx],
                ]
            )
        elif u1 > ymax:
            new_image = np.vstack(
                [
                    self.image[u0:, sx],
                    np.zeros((u1 - ymax, sx.stop - sx.start), dtype="bool"),
                ]
            )
        else:
            new_image = self.image[u0:u1, sx]

        return new_image


class Page:
    """The class for the pages of the treated scanned  documents."""

    @staticmethod
    def _black_pixels_to_front(img: np.ndarray) -> np.ndarray:
        """Converts image to a boolean array where black pixels become True.

        Args:
            img (np.ndarray): The image to treat.

        Returns :
            A np.ndarray of 'bool' type where the black pixels of img are
            converted to True.
        """
        if img.ndim == 2:
            median = np.quantile(img, 0.5)
            the_min = np.amin(img)
            the_max = np.amax(img)
            if median > the_max / 2:
                output = img < the_min + (the_max - the_min) / 5
            else:
                output = img > the_max - (the_max - the_min) / 5
            return output
        else:
            raise ValueError("A 2-dimensional np.ndarray was expected.")

    def __init__(
        self,
        img: np.ndarray,
        char_height: int,
        char_thickness: int,
        char_width: int,
    ):
        """Instantiate Page Object.

        Args:
            img (np.ndarray): The image of a scanned handwritten text.
            char_height (int): An estimate of the height of the character,
                measured in pixels.
                The value should be around the height of an upper case
                character.
            char_thickness (int): The thickness of the characters, in
                pixels.
            char_width (int): The approximate width of a lower case
                character, in pixels.
        """
        self.image = Page._black_pixels_to_front(img)
        self._line_heights = None
        self._lines = None
        self.char_height = char_height
        self.char_thickness = char_thickness
        self.char_width = char_width

    def extract(self, sub_image: np.ndarray) -> PageExtract:
        """Give the PageExtract of self associated to sub_image.

        Args:
            sub_image (ndarray): an ndarray with the same shape as self.

        Returns:
            PageExtract: The bounding box of the front (non zero )
            pixels of sub_image as a PageExtract.
        """
        s = bounding_box_slices(sub_image != 0)
        sy, sx = s
        return PageExtract(
            sub_image[sy, sx],
            self.char_height,
            self.char_thickness,
            self.char_width,
            sy.start,
            sx.start,
            self.image.shape,
        )

    def extract_line(self, sub_image: np.ndarray, line_height: int) -> Line:
        """Convert subimage and line height to a line.

        Args:
            sub_image (ndarray): Has the same shape as self and has
            front pixels corresponding to a given line.
            line_height (int): A y coordinate in self for the
            line's height.

        Returns:
            Line: The he resulting line in the bounding
            box of sub_image.
        """
        page_extract = self.extract(sub_image)
        return Line(page_extract, line_height - page_extract.corner_y)

    def line_heights(
        self,
        delta: int | None = None,
        smoothing_window_width: int | None = None,
        noise_removal: int | None = None,
    ) -> list[int]:
        """Checks if the line heights are already computed.
        If yes, these are returned. Else they are computed.
        If one of the optional parameter values is specified,
        the line heights are recomputed with the prescribed parameter
        values.

        The algorithm is based on finding the local maxima of the function
        height->pixel count of the 1-pixel-high vertical line with this
        height. However, before analyzing the function, it is smoothened
        by replacing the function's values by homogeneous means over
        intervals of width 'smoothing_window_width'.

        Some local maxima are considerd irrelevant and discarded
        if the value fo the smoothened function is not over
        'noise_removal' at these local maxima.

        Actually, The notion of a local maximum may be discretized in
        various manners. Ours is the following, defined in terms of the
        parameter 'delta': a point p is considered
        a local maximum if the values at p+delta and p-delta are smaller
        or equal than the value at p.

        Args:
            delta (int | None, optional): Defines the used notion of local
                maximum. Defaults to None. If None is passed, a value
                is fixed automatically.
            smoothing_window_width (int | None, optional): Defines the
                sized of the used smoothing window. Defaults to None.
                If None is passed, a value is fixed automatically.
            noise_removal (int | None, optional): Define the pixel number
                threshold under which the line is ignored. Defaults to
                None. If None is passed, n value is fixed automatically.

        Returns:
            list[int]: The relevant line heights.
        """
        if (
            (self._line_heights is None)
            or delta is not None
            or smoothing_window_width is not None
            or noise_removal is not None
        ):
            heights = self._find_line_heights(
                delta=delta,
                smoothing_window_width=smoothing_window_width,
                noise_removal=noise_removal,
            )
            self._line_heights = heights

        return self._line_heights

    def _find_line_heights(
        self, delta=None, smoothing_window_width=None, noise_removal=None
    ):
        """Calculates the line heights in terms of certain
        options, see line_heights docstring."""
        if delta is None:
            delta = self.char_height // 2
        if smoothing_window_width is None:
            smoothing_window_width = self.char_height
        if noise_removal is None:
            noise_removal = self.char_thickness // 3
        img = self.image
        v_rep = np.sum((img).astype("int32"), axis=1)
        v_rep_smooth = ndimage.convolve(v_rep, np.ones(smoothing_window_width))
        line_heights = []
        interval_first_point = []
        interval_width = []
        delta = self.char_height // 2
        # we obtain intervals of contiguous local max, we record these
        # intervals and take their centers
        on_top = False
        for i in range(v_rep.shape[0]):
            if (
                (
                    i in range(delta, v_rep.shape[0] - delta)
                    and v_rep_smooth[i] >= v_rep_smooth[i + delta]
                    and v_rep_smooth[i] >= v_rep_smooth[i - delta]
                )
                or (
                    i in range(delta)
                    and v_rep_smooth[i] >= v_rep_smooth[i + delta]
                )
                or (
                    i in range(v_rep.shape[0] - delta + 1, v_rep.shape[0])
                    and v_rep_smooth[i] >= v_rep_smooth[i - delta]
                )
            ):
                if on_top:
                    interval_width[-1] += 1
                else:
                    interval_first_point.append(i)
                    interval_width.append(1)
                    on_top = True

            else:
                on_top = False

        line_heights = [
            interval_first_point[i] + interval_width[i] // 2
            for i in range(len(interval_width))
        ]
        filtered_line_heights = []
        for h in line_heights:
            if v_rep_smooth[h] > noise_removal:
                filtered_line_heights.append(h)

        return filtered_line_heights

    def lines_number(self):
        """The number of detected lines in self."""
        return len(self.line_heights())

    @staticmethod
    def what_line_heights(
        subimage: np.ndimage, line_heights: list[int], line_thickness: int
    ) -> list[int]:
        """Determine the line heights relevant to a subimage of a Page.

            Returns the elements h of line_heights such that a thickline
            of thickness line_thickness centered at height h meets some
            non  0 pixel of subimage.

        Args:
            subimage (np.ndimage): The subimage (a grayscale or booelan
                image)
            line_heights (list[int]): The line heights to be tested
            line_thickness (int): The thickness parameter of the test.

        Returns:
            list[int]: List of heights that pass the test.
        """
        thick_lines = horizontal(
            subimage, line_heights, line_thickness, as_list=True
        )
        return [
            line_heights[i]
            for i in range(len(line_heights))
            if np.any(thick_lines[i] & subimage)
        ]

    # we will need to separate lines related by an accidental symbol going from one line to another
    def _find_bridges(
        self, bridged_lines, line_heights=None, line_thickness=None
    ):
        if line_heights is None:
            line_heights = self.line_heights()
        if line_thickness is None:
            line_thickness = self.char_height // 2
        bridged_lines_heights = Page.what_line_heights(
            bridged_lines, line_heights, line_thickness
        )
        bridges = []

        for i in range(len(bridged_lines_heights) - 1):
            h1 = bridged_lines_heights[i]
            h2 = bridged_lines_heights[i + 1]
            cut1 = h1 + line_thickness // 2
            cut2 = h2 - line_thickness // 2
            white_top = np.zeros(bridged_lines[:cut1].shape, dtype="bool")
            white_bottom = np.zeros(bridged_lines[cut2:].shape, dtype="bool")

            in_between = bridged_lines[cut1:cut2]
            labels = skimage.measure.label(in_between)
            for label in range(1, np.amax(labels) + 1):
                v_rep = np.sum(labels == label, axis=1)

                if v_rep[0] * v_rep[-1] != 0:
                    bridges.append(
                        np.vstack([white_top, labels == label, white_bottom])
                    )
        return bridges
        # we observe that bridged_lines need not be necessarily bridged,
        #  the returned list can as well be empty.

    def _bridge_cutting_shape(self, bridge, thickness=None):
        if thickness is None:
            thickness = 5 * self.char_thickness // 4
        v_rep = np.sum(bridge, axis=1)
        region = np.where(v_rep > 0)[0]
        top = region[0]
        bottom = region[-1]
        center = (bottom + top) // 2
        thin = (v_rep < thickness) & (v_rep > 0)
        # we find the right place to cut by successive approximations
        right_place = center
        if np.any(thin):
            min_thickness = np.amin(v_rep[top:bottom])
            candidates = np.where(v_rep == min_thickness)[0]
            right_place = candidates[0]
            for i in candidates[1:]:
                if (i - center) ** 2 < (right_place - center) ** 2:
                    right_place = i
        right_place_line = horizontal(bridge, [right_place], thickness=2)
        shape = right_place_line & bridge

        return shape

    def lines(self, line_thickness: int | None = None) -> list[Lines]:
        """Separate the Lines of self.

        The used algorithm separates the lines in the following manners.
        For each height in self.line_heights(), imagine a thick horizontal
        line centerd at this height, whith thickness given by the
        'line_thickness' parameter.

        If no connected component of
        self.image meets two of these thick lines, one wishes to put
        together all the components that touch a given thickline to form
        the basic line attributed to the corresponding height. Afterwards,
        the still non attributed components are considered as satellites
        and attached to some basic lines under certain criteria.

        When some connected component of self.image meets several of
        these thick lines, a cutting procedure is used to separate the
        various batches.


        Args:
            line_thickness (int| None, optional): The thickness parameter
                of the algorithm. Defaults to None. If None is passed,
                this parameter is set automatically.


        Returns:
            list[Lines]: The list of the constructed Lines, one per element
                in self.line_height.
        """
        if self._lines is None or line_thickness is not None:
            basic_lines, satellites = self._find_basic_lines(
                line_thickness=line_thickness
            )

            complete_lines = self._line_constellations(
                basic_lines,
                satellites,
                line_thickness=line_thickness,
            )
            self._lines = [
                self.extract_line(complete_lines[i], self.line_heights()[i])
                for i in range(len(complete_lines))
            ]
        return self._lines

    def _find_basic_lines(self, line_thickness=None):
        if line_thickness is None:
            line_thickness = self.char_height // 2
        img = self.image
        line_heights = self.line_heights()
        h_lines = horizontal(img, line_heights, line_thickness)

        # components labeling
        to_be_labeled = h_lines | img
        labels = skimage.measure.label(to_be_labeled)

        # components classification, by size
        big_enough = []
        fat_lines = []
        for i in range(1, np.amax(labels) + 1):
            size = comp_size(labels, i)
            if size > self.char_thickness**2 // 4:
                big_enough.append(i)
            if size >= self.char_height // 8 * img.shape[1]:
                fat_lines.append(i)
        # print("total number of components:", len(big_enough) + len(fat_lines))
        # big_enough = [
        #    i
        #    for i in range(1, np.amax(labels) + 1)
        #    if check_comp_size(
        #        labels, i, threshold=self.char_thickness**2 // 4
        #    )
        # ]
        # fat_lines = [
        #    i
        #    for i in big_enough
        #    if check_comp_size(
        #        labels, i, threshold=self.char_height // 8 * img.shape[1]
        #    )
        # ]

        tentative_lines = [
            ((labels == i) * img).astype("bool") for i in fat_lines
        ]
        satellites = [
            ((labels == i) * img).astype("bool")
            for i in big_enough
            if not i in fat_lines
        ]

        the_satellites = sum(satellites) > 0

        # we find bridges in the_tentative_lines and cut them.
        shapes = np.zeros(img.shape, dtype="bool")
        for t_line in tentative_lines:
            bridges = self._find_bridges(t_line, line_thickness=line_thickness)
            for bridge in bridges:
                shape = self._bridge_cutting_shape(bridge)
                shapes = shapes | shape

        the_tentative_lines = sum(tentative_lines) > 0
        the_basic_lines = the_tentative_lines ^ shapes

        # we then use the same labeling strategy with the_basic_lines
        to_be_labeled = h_lines | the_basic_lines
        labels = skimage.measure.label(to_be_labeled)
        basic_lines = [
            (labels == i) * img for i in range(1, np.amax(labels) + 1)
        ]
        # The basic lines seem to be ordered with respect to their line
        # heights but this feature is not clearly guaranteed
        # by skimage.measure.label
        basic_lines.sort(
            key=lambda line: Page.what_line_heights(
                line, line_heights, line_thickness
            )[0]
        )

        return [basic_lines, the_satellites]

    # we try to attribute the satellites to the right lines

    def _line_constellations(
        self,
        basic_lines,
        the_satellites,
        line_thickness=None,
        fatness=None,
        verbose=False,
    ):
        if line_thickness is None:
            line_thickness = self.char_height // 2
        if fatness is None:
            fatness = 3 * self.char_height // 4
        thick_lines = horizontal(
            the_satellites, self.line_heights(), line_thickness, as_list=True
        )

        used_sats = set({})
        complete_lines = []

        footprint = np.array(
            [[True] * (2 * (fatness // 2) + 1)] * (fatness + 1)
            + [[True] * (2 * (fatness // 2) + 1)] * (fatness // 3)
            + [[False] * (2 * (fatness // 2) + 1)] * (fatness - fatness // 3)
        )
        for u in range(len(basic_lines)):
            basic_line = basic_lines[u]
            thick_line = thick_lines[u]
            mask = basic_line | the_satellites
            # Tried to enhance perf using grey dilation instead of binary dilation
            # it was actually worst.
            # The hand_made_binary_dilation used below has actually
            # better performance than the one of skimage, provided the binary
            # image is sparse enough, namely has less than 2% of front pixels.
            fattened_line = hand_made_binary_dilation(
                (basic_line | thick_line), footprint
            )
            to_be_labeled = fattened_line | the_satellites
            labels = skimage.measure.label(to_be_labeled)
            # We keep track of the various satellites labelling in complete_line.
            # The basic_line is labelled with 1.
            the_labels = np.unique(fattened_line.astype("uint16") * labels)
            complete_line = sum(
                [
                    ((labels == the_label) * mask).astype("uint16") * the_label
                    for the_label in the_labels
                    if the_label != 0
                ]
            )
            i, j = _front_pixel(basic_line)
            basic_line_label = complete_line[i, j]
            complete_line[complete_line == 1] = basic_line_label
            complete_line[basic_line] = 1
            complete_lines.append(complete_line)

        return complete_lines


class MatchedGlyph:
    """The class of matched glyphs i.e.  pairs (glyph,string).

    Here we mainly consider glyphs as Words formed by a single handwritten
    character. However, if needed we might accept also the generalized concept
    of the iamge representation of a piece of text that can be extracted
    from a page as a Word object."""

    def __init__(self, glyph: Word, string: str):
        """Instantiate MatchedGlyph, matching pairs (glyph,string).

        Args:
            glyph (Word): a Word object formed around the image representation
                of a string
            string (str): The string represented in the Word.
        """
        self.glyph = glyph
        self.string = string


class WordMatcher:
    """A class to match glyphs with strings."""

    def __init__(self, word: Word, text: str, punctuation: str | None = None):
        """Instantiate the glyph matcher.

        Args:
            word (Word): a suitably cut Word,
                that may include punctuation.
            text (str): the text to be matched with the Word
                punctuation (str, optional): The set of characters to be
                considered as punctuation. If not passed, it will be deduced
                from the punctuation variable of the string package.
        """

        self.word = word
        self.text = text
        if isinstance(punctuation, str):
            self.punctuation = punctuation
        else:
            self.punctuation = "\u2014" + " " + stringlib.punctuation

    # We once imagined to give a special role to punctuation/any family of
    # distinguished characters when matching,
    # whence the writing of the methods punctuation_split and has_punctuation.
    # We do not use this strategy for the moment.

    def punctuation_split(self) -> list[str]:
        """Separate self.text in word blocks and punctuation blocks.

        Returns:
            list[str]: the list of its blocks, by order of appearance.
                The list always starts with a non punctuation block,
                possibly empty, so that odd index blocks correspond
                exactly to punctuation blocks.
        """
        output = []
        is_punctuation_block = False
        current_block = ""
        for char in self.text:
            is_changing = (char in self.punctuation) != is_punctuation_block
            if is_changing:
                output.append(current_block)
                is_punctuation_block ^= True
                current_block = char
            else:
                current_block += char
        if current_block:
            output.append(current_block)
        return output

    @property
    def has_punctuation(self) -> bool:
        """Decide if self.text contains punctuation.

        Returns:
            bool: True if self.text contains punctuation, False otherwise.
        """
        for c in self.text:
            if c in self.punctuation:
                return True
        return False

    def match(self) -> list[MatchedGlyph]:
        """Try to match self.Word with self.text as MatchedGlyphs.

        Raises:
            ValueError: "The number of components in self.Word do not
                match self.text's length."

        Returns:
            list[MatchedGlyph]: The deduced list of MatchedGlyphs.
        """
        components = self.word.find_word_components(sat_margin=0)

        if len(components) != len(self.text):
            raise ValueError(
                "The number of components in self.Word do not "
                + "match self.text's length."
            )
        output = []
        for i, c in enumerate(self.text):
            output.append(MatchedGlyph(components[i], c))
        return output


class CutInfo:
    """The class for information on tentative cuts, good or wrong ones."""

    def __init__(
        self,
        subword: Word,
        cutting_shape_chars: tuple[int, int, int],
        quality: bool,
    ):
        """Instantiate CutInfo

        Args:
            subword (Word): the surrounding Word of a tentative cut,
                extracted between the two neighboring tentative cuts.
                (The end and beginning of the parsed Word are considered
                neighboring tentative cuts in the definition of the
                surrounding Word)
            cutting_shape_chars (tuple[int,int,int]): The cutting shape
                described as a tuple (y,x,height) where (y,x) are the
                coordinates of the center of the cutting vertical segment
                and height is its height.
            quality (bool): True if the user accepted the cut, False
                otherwise.
        """

        if (
            cutting_shape_chars[0] >= subword.image.shape[0]
            or cutting_shape_chars[1] >= subword.image.shape[1]
            or cutting_shape_chars[2] > subword.image.shape[0]
        ):
            ValueError(
                "The passed Word and cutting shape chars are not compatible."
            )
        subword = subword.word_part(_sanitized_labels(subword.image))
        self.subword = subword
        self.cutting_shape_chars = cutting_shape_chars
        self.quality = quality


class CutParser:
    """The class of objects that parse the user's cut choice."""

    def __init__(
        self,
        word: Word,
        cutting_shapes: list[np.ndarray],
        cut_positions: list[tuple[int, int]],
        cut_choices: list[bool],
    ):
        """Instantiate CutParser.

        Args:
            word (Word): The Word beeing cut.
            cutting_shapes (list[np.ndarray]): The cutting shapes as
                boolean images of the same shape as Word.image.

            cut_positions (list[tuple[int,int]]): The coordinate tuples
                of the centers of the cutting shapes in the form (y,x).
            cut_choices (list[bool]): The information of which
                cutting_shapes where chosen. True for chosen, False
                otherwise.
        """
        # basic data validation
        reference_shape = word.image.shape
        if any(
            [
                cutting_shape.shape != reference_shape
                for cutting_shape in cutting_shapes
            ]
        ):
            raise ValueError(
                "Your Word's image does not have the same"
                + " shape as every image in cutting_shapes"
            )
        reference_length = len(cutting_shapes)
        if (
            len(cut_positions) != reference_length
            or len(cut_choices) != reference_length
        ):
            raise ValueError(
                "All the passed lists do not have the same length."
            )

        self.word = word
        self.cutting_shapes = cutting_shapes
        self.cut_positions = cut_positions
        self.cut_choices = cut_choices

        if cutting_shapes:
            shape_vert_proj = np.where(cutting_shapes[0])[0]
            shape_height = (
                np.amax(shape_vert_proj) - np.amin(shape_vert_proj) + 1
            )
            self.shape_height = shape_height

    def subword_from_cutting_position(self, i: int) -> Word:
        """Find the subword between the neighboring tentative cuts.

        Args:
            i (int): The index of the considered cut in
                self.cut_positions.

        Returns:
            Word: The sought Word, extracted from self.word.
        """
        if i == 0:
            x1 = 0
        else:
            x1 = self.cut_positions[i - 1][1]
        if i == len(self.cut_positions) - 1:
            x2 = self.word.image.shape[1]
        else:
            x2 = self.cut_positions[i + 1][1]
        return self.word.word_part(self.word.image[:, x1:x2])

    def get_cut_infos(self) -> list[CutInfo]:
        """Return the list of CutInfos from our cut choices.

        Returns:
            list[CutInfo]: The list of CutInfo objects associated
                cutting self.Word according to self.cut_choices among
                self.cutting_shapes.
        """
        cut_infos = []
        for i in range(len(self.cut_positions)):
            previous_x = 0 if i == 0 else self.cut_positions[i - 1][1]
            relative_cut_chars = (
                self.cut_positions[i][0],
                self.cut_positions[i][1] - previous_x,
            ) + (self.shape_height,)
            subword = self.subword_from_cutting_position(i)
            cut_infos.append(
                CutInfo(subword, relative_cut_chars, self.cut_choices[i])
            )
        return cut_infos
