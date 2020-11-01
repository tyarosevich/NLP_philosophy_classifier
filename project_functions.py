#%% Declare functions to be used throughout analysis
# Some iteration code taken from https://stackoverflow.com/questions/26494211/extracting-text-from-a-pdf-file-using-pdfminer-in-python/26495057#26495057


#%% Imports
import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
from project_functions import *

# Function that takes the path of a pdf and returns it as a string object.
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
        test = page
        interpreter.process_page(page)

    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    return text

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

# Function to generate all the raw lists of strings for each pdf. Significant Processing time.
def process_paths(path1, path2):
    corpus1, label_list1  = pdf_to_strings(path1)
    corpus2, label_list2  = pdf_to_strings(path2)
    return corpus1, label_list1, corpus2, label_list2

# A function to stem a list of strings composed of pdfs. Does so iteratively, then joins them back
# into single strings and returns a list of strings like the argument.
def list_stemmer(doc_list):
    n = len(doc_list)
    stemmed_text_list = []

    # Initialize the stemmer
    stemmer = nltk.SnowballStemmer("english")

    for i in range(0, n):
        # According to stackexchange discussions, a list comprehension is much faster for this task
        # than a loop.
        stemmed_text = ' '.join(stemmer.stem(token) for token in nltk.word_tokenize(doc_list[i]))
        stemmed_text_list.append(stemmed_text)
    return stemmed_text_list

def list_lemmatizer(doc_list):
    from nltk.stem import WordNetLemmatizer
    n = len(doc_list)
    lemmad_text_list = []

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    for i in range(0, n):
        # According to stackexchange discussions, a list comprehension is much faster for this task
        # than a loop.
        lemmad_text = ' '.join(lemmatizer.lemmatize(token) for token in nltk.word_tokenize(doc_list[i]))
        lemmad_text_list.append(lemmad_text)
    return lemmad_text_list

# Removes numbers from the argument, which should be a list of strings.
def num_removal(doc_list):
    for i, list in enumerate(doc_list):
        doc_list[i] = re.sub(r'\d+', '', list)

# Removes most names from the argument, which should be a list of strings
def name_removal(doc_list):
    for i, list in enumerate(doc_list):
        tagged_string = nltk.tag.pos_tag(list.split())
        new_string = [word for word,tag in tagged_string if tag != 'NNP' and tag != 'NNPS']
        doc_list[i] = ' '.join(new_string)

# Load the desired data. Arguments are the path and whether raw, stemmed, or lemmatized data
# Is desired. Returns part1_data, part1_labels, part2_data, part2_labels. Code has no error
# control, type should be "lemmed", "stemmed" or "raw
def load_data(path, type):
    import pickle
    with open(path + "/part1_data_" + type, 'rb') as f:
        part1_data = pickle.load(f)

    with open(path + "/part1_labels", 'rb') as f:
        part1_labels = pickle.load(f)

    with open(path + "/part2_data_" + type, 'rb') as f:
        part2_data = pickle.load(f)

    with open(path + "/part2_labels", 'rb') as f:
        part2_labels = pickle.load(f)
    return part1_data, part1_labels, part2_data, part2_labels;

# Save files based on type. Again, no error control, same type as the load function
# must be used.
def save_data(path, type, corpus1, label_list1, corpus2, label_list2):
    with open(path + "/part1_data_" + type, 'rb') as f:
        pickle.dump(corpus1, f)

    with open(path + "/part1_labels", 'rb') as f:
        pickle.dump(label_list1, f)

    with open(path + "/part2_data_" + type, 'rb') as f:
        pickle.dump(corpus2, f)

    with open(path + "/part2_labels", 'rb') as f:
        pickle.dump(label_list2, f)

