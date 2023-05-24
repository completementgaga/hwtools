# Ideas

1. ~~restore the horizontal bounding-boxing for the extracted characters.~~

1. ~~use thinner cutting shape, several examples showed too big cutting shapes lead to cut almost vertical lines in two parts, which is bad.~~

1. ~~change how neighborhing cutting shapes are put together, so that the resulting cut choice appears in the middle of the pack.~~

1. ~~split tolerance of event_detected as pixel_tolerance and height_tolerance and fine tune these.~~

1. ~~refine pixel events definition~~

1. ~~add a condition on the size of the created components when checking a candidate cutting shape actually cuts the main component in a window (see keyword arg window size or so to find this part of the code)~~ 

1. ~~add (or fix the existing one) a filter to remove small connected components (say smaller thant char_thickness**2//4)~~ seems to be already ok

1. ~~In in_between_comp, it is implicit that the basic line is connected, it seems that not every basic line~~
    ~~of a word component is connected (issue with first occurence of Fleurissoire).~~ This was an error in find_word_components

1. ~~do not take that much satellites from the neighborhing letters, accept only satellites that meet a relatively small neighborhood of the letter, maybe obtained by adding a margin of char_width//3, not more.~~   we set sat_margin to 0

1.  ~~postpone checking of new components sizes from is_critical to the end of find_word_cutting_shapes_in_component.~~

1. ~~add skip button in the gui, to skip word.~~

1. ~~transfer print('blah') of trainer to my_ui.say('blah')~~

1. ~~add a storing style instance attribute in DataManager.~~

1. ~~describe the list of tunable parameters and make them easily tunable through the API~~

1. ~~Let the place for logs be defined in config.py~~


1. ~~Transfer initial document manipulation from ui_manager to data_manager (SOLID)~~

1. ~~Fix ui abstract class to be consistant with our first ui.~~

1. ~~End up documentation of all APIs -- currently at the beginning of class Page in parser.~~

1. ~~replace prints in parser by suitable logging.~~

1. ~~Check comments consistency with current code~~ skipped, because I don't manage to read them anymore.
 
1. Release prior to finding the right metrics and keep the exploitation tricks secret, or

1. ~~or share limitedly not using github, in a private 'portfolio'~~ chose to release on github with MongoDB's SSPL

1. add a data checker that allows simple exclusion of wrongly parsed data.


# BUG FIX
1. ~~there is an issue with pixel events, always jumping from 0 to 4/5~~
1. ~~check whether the cut_comp passed to check event is well computed~~ seems ok
1. ~~check the pixel_tolerance and height_tolerance are always passed in the right order
    to ensure this use keyword version of these arguments EVERYWHERE.~~

1. ~~Some nice cuts seem to have disappeared after using the in_between_component.
    Should it be used only for pixel events ? eg s'attennuaient is not cur correctly anymore.~~ removed side sensitivity in in_between_comp
