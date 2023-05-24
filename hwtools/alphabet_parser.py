import ui_manager
import parser
import numpy as np
import data_manager
import os


my_ui = ui_manager.LocalUi()

author = my_ui.get_author()

full_document_path = my_ui.get_document(author)

char_width = my_ui.get_char_width()

char_height = my_ui.get_char_height()

char_thickness = my_ui.get_char_thickness()

# lines below to get an alphabet
# path='/home/gael/Documents/jupy-notebooks/tensorflow/handwriting_recognition/alphabet/dejavu_alphabet.png'
# text_path='/home/gael/Documents/jupy-notebooks/tensorflow/handwriting_recognition/alphabet/transcripted_alphabet.txt'
# author='the_alphabet'
# end of alphabet specific code

full_directory_path = os.sep.join(full_document_path.split(os.sep)[:-1])
text_path = os.path.join(full_directory_path, "transcription.txt")
storing_directory = full_directory_path
csv_path = os.path.join(full_directory_path, "metadata.csv")

with open(text_path) as f:
    lines = f.readlines()
# lengths=[len(line.replace(" ", "").replace("\n", "")) for line in lines]
text = " ".join(lines)
text = text.replace(" ", "").replace("\n", "")


page1 = parser.Page(
    full_document_path, char_height, char_thickness, char_width
)



lines = page1.lines()
print("The line heights are: ", page1.line_heights())
char_count = 0



my_data_manager = data_manager.DataManager(storing_directory, csv_path)
for i, line in enumerate(lines):
    # We are a bit far from a true hwr problem, the tuning below is necessary 
    # to get the smallest characters (e.g. ¨ and `) of our alphabet.
    # For a handwritten alphabet, this might be an issue to set min_width=0,
    # as it entails no elimination of noise at this step.
    # However, if we decide to pass the diacritical marks in context, the
    # thinest elements of the alphabet will get fatter, so that min_width 
    # could be set to a reasonable >0 value, maybe its default value, in order 
    # to eliminate noise.
    words = line.words(min_width=0,max_gap=2*char_thickness)
    print("line number: ", i)
    j = 0
    for j, word in enumerate(words):
        file_name = "line_" + str(i) + "_char" + str(j)
        to_store = data_manager.MatchedGlyph(word, text[char_count])
        char_count += 1
        my_data_manager.store_glyph(to_store, name=file_name)
    # print("Finished: ",j+1,' elements parsed out of',lengths[i],'.\n\n')



print("Finished: ",char_count,' elements parsed out of',len(text),'.')

