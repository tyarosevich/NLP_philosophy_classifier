# Journal list:
# continental: philosophy and phenomenological research, european journal of philosophy, continental philosophy review
# analytic: mind, nous, the philosophical review, analysis, the journal of philosophy,
# good: mind, philosophy and phenomenological research, the philosophical review, the APA journal, the australasion journal of phil,
# bad: Episteme, dianoia, sapere aude, stance, aporia, ergo



#%% Test code for reading in a PDF file and converting it to a string using the pdfdminer package
# Code taken from https://stackoverflow.com/questions/26494211/extracting-text-from-a-pdf-file-using-pdfminer-in-python/26495057#26495057

import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    # Skips the first page of the pdf file since it contains data that can be considered noise.
    pages = PDFPage.get_pages(fp, pagenos, maxpages=maxpages,password=password,caching=caching, check_extractable=True)
    iter_pages = iter(pages)
    next(iter_pages)
    for page in iter_pages:
        interpreter.process_page(page)

    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    return text
#Desktop
# path = "C:/Users/zennsunni/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/bad_papers/7 Hood Politics.pdf"

# Laptop
path = "C:/Users/tyran/Dropbox/Journal Papers/Braidotti - NomadicEthics.pdf"


test = convert_pdf_to_txt(path)

#%% Test code to convert to a bag of words
# Code taken from https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords


documentA = test

# # Splits the string into a list
# bagOfWordsA = documentA.split(' ')
#
# # Converts to a set to get rid of duplicate words
# uniqueWords = set(bagOfWordsA)
#
# # Creates a dictionary of words and their frequency
# numOfWordsA = dict.fromkeys(uniqueWords, 0)
# for word in bagOfWordsA:
#     numOfWordsA[word] += 1

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([documentA])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

#%% So it seems we want a list of strings where each string is a document. Our function already spits out
# a string so this is easy. can just iterate through a folder. Then as we import, we'd save the label for that document.
# So I think ultimately what we'll want is to follow the format we found. A list of the documents as big strings will
# be passed to the vectorizer/fit function.along with the list of category codes ( the dictionary is separate and
# used to unzip those codes later). Stop words and lowercasing can be built into the vectorizer/fit function.
# Careful consideration of the parameters here is important. THis builds the feature vectors that are then fed into
# The KNN classifier.
# Note: We will WANT strong inverse frequency determination. We will also want to look at most correllated unigrams
# and bigrams, perhaps trigrams. Would be wise to use pickles to export files as well.

#%% Iterate through each folder and create the data set

# This dictionary gives the folder names as keys to the artists inside them.
label_index1 = {"continental": 0, "analytic":1}
label_index2 = {"good": 0, "bad": 1}
# A function that iterates through a given root directory and extracts all pdf files as a list of strings.
def pdf_to_strings(path):
    rootdir = path

    corpus_raw = []
    index = 0
    label_list = []


    # Iterates through the base folder with os.walk, which goes through each subfolder, with
    # iterate variables for the files, the subdirectories and the directories.
    for subdir, dirs, files in os.walk(rootdir):

        # Iterates through each file.
        for file in files:

            # Gets the file path to pass to the extraction function.
            file_path = os.path.join(subdir, file)
            # Appends the name of the current folder, which is the label for the pdf
            label_list.append(os.path.basename(os.path.normpath(subdir)))
            corpus_raw.append(convert_pdf_to_txt(file_path))
    return corpus_raw, label_list;

#%% Call the function and return a list of the pdfs. Significant Processing time.

path1 = "C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part1"
path2 = "C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part2"

corpus1, label_list1  = pdf_to_strings(path1)
corpus2, label_list2  = pdf_to_strings(path2)

#%% Pickle the raw data
import pickle
with open("C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part1_raw_data", 'wb') as f:
    pickle.dump(corpus1, f)

with open("C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part1_labels", 'wb') as f:
    pickle.dump(label_list1, f)

with open("C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part2_raw_data", 'wb') as f:
    pickle.dump(corpus2, f)

with open("C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part2_labels", 'wb') as f:
    pickle.dump(label_list2, f)

#%% Load the data
import pickle

# Dicts for converting from string to numerical labels
label_index1 = {"continental": 0, "analytic":1}
label_index2 = {"good": 0, "bad": 1}



with open("C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part1_raw_data", 'rb') as f:
    part1_data_raw = pickle.load(f)

with open("C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part1_labels", 'rb') as f:
    part1_labels_str = pickle.load(f)

with open("C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part2_raw_data", 'rb') as f:
    part2_data_raw = pickle.load(f)

with open("C:/Users/tyran/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part2_labels", 'rb') as f:
    part2_labels_str = pickle.load(f)