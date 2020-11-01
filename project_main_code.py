# Journal list:
# continental: philosophy and phenomenological research, european journal of philosophy, continental philosophy review
# analytic: mind, nous, the philosophical review, analysis, the journal of philosophy,
# good: mind, philosophy and phenomenological research, the philosophical review, the APA journal, the australasion journal of phil,
# bad: Episteme, dianoia, sapere aude, stance, aporia, ergo

# Notes for further work:
# When we implement a NN to compare, try using transfer learning from the continental/analytic
# set to the good/bad set. Actually this is rather naive. Use transfer learning from some open source
# project and then train these. It would bea  good exercise.
# Alternatively it might be cool to make a "talk like a philosopher" chat bot. Note that some
# very advanced, recent tunable models are available: BERT (Google), ELMo (Allen), GPT-2 (Open AI)

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
import matplotlib.pyplot as plt
import seaborn as sns

#%% Functions for the project

# Perform a reduced SVD of the data for Part 1 and plot the singular values on a standard and semi-log axis.
def svd_plot(data):
    A1 = data

    U, S, V = np.linalg.svd(A1, full_matrices=False)
    x = np.linspace(1, 50, 50)

    # Plots the first 50 singular values.
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x, S[0:50], 'r-o')
    ax1.set_xlabel('$\sigma_j$')
    ax1.set_ylabel('$\sigma$ value')

    ax2.semilogy(x, S[0:50], 'k-o', )
    plt.rc('text', usetex=True)
    ax2.set_xlabel('$\sigma_j$')
    ax2.set_ylabel('log of $\sigma$ value')
    plt.show()

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

#%% Download nltk resources and load the various functions written for the project from file. Only really
# needs to be run once.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#%% Get the raw strings and pickle the data
path1 = "C:/Users/zennsunni/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part1"
path2 = "C:/Users/zennsunni/Dropbox/School Stuff/Winter 2020/AMATH_582/Project/part2"
corpus1, label_list1, corpus2, label_list2 = process_paths(path1, path2)

path = "C:/Users/zennsunni/Dropbox/School Stuff/Winter 2020/AMATH_582/Project"
type = "raw"
save_data(path, type, corpus1, label_list1, corpus2, label_list2)

#%% Stem the data and save these files separately

corpus1 = list_stemmer(part1_data_raw)
corpus2 = list_stemmer(part2_data_raw)

save_data(path, type, corpus1, label_list1, corpus2, label_list2)

#%% Lemmatize the data and save these files separately

corpus1 = list_lemmatizer(part1_data_raw)
corpus2 = list_lemmatizer(part2_data_raw)

save_data(path, type, corpus1, label_list1, corpus2, label_list2)

#%% Load the desired data
path = "C:/Users/zennsunni/Dropbox/School Stuff/Winter 2020/AMATH_582/Project"
type = "lemmed"
part1_data, part1_labels, part2_data, part2_labels = load_data(path, type)
#%% Remove numbers and most names from date before setup
num_removal(part1_data)
num_removal(part2_data)

name_removal(part1_data)
name_removal(part2_data)

#%% Setup and Classification
from sklearn.feature_extraction import text

data1 = part1_data
data2 = part2_data

# This dictionary gives the folder names as keys to the artists inside them.
label_index1 = {"continental": 0, "analytic":1}
label_index2 = {"good": 0, "bad": 1}

# Create numerical labels
part1_labels_int = [label_index1[q] for q in part1_labels]
part2_labels_int = [label_index2[q] for q in part2_labels]

# Put the data in a data frame
prep_dict1 = {'part1_data': data1, 'part1_labels':part1_labels, 'part1_id': part1_labels_int}
prep_dict2 = {'part2_data': data2, 'part2_labels':part2_labels, 'part2_id': part2_labels_int, }
df1 = pd.DataFrame(prep_dict1)
df2 = pd.DataFrame(prep_dict2)

# Split the corpus up into randomized training/testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(df1['part1_data'], df1['part1_id'], test_size=0.15, random_state=8)
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2['part2_data'], df2['part2_id'], test_size=0.15, random_state=8)

#%%

# Parameter selection for vectorization of the data. We will look at unigrams, bigrams, and trigrams. We will
# Also ignore words that appear in less than 5% of the documents. Max features is set very high, since a few odd words
# Might be very correllated with a particular label, and philosophy papers tend to use the full spectrum of the
# English Language. This might have to be tuned up to a very high number.
ngram_range = (1,2)
min_df = 1
max_df = 25
max_features = 2000

# Additional stop words, which get added to sklearn's stop word list.
stop_list = ["Authors", "Published", "Permissions", "journal", "journals", "reserved", "Inc", "Philosophical", "Review",
             "org", "This", "content", "downloaded", "https", "All", "Oxford", "New", "York", "university", "University", "Vol", "http", "doi", "Press",
             "And", "org", "UTC", "Mar", "ed", "downloaded", "Sun", "Mon", "Tues", "Wed", "Thur", "Fri", "jstor", "Cornell",
             "DOI", "European", "Continental", "Springer", "Sons", "Wiley", "wileyonlinelibrary","January", "February",
             "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Stance",
             "Dianoia", "Undergraduate", "Boston", "College", "Aporia", "Ergo", "no." "no", "Phenomenological", "Research",
             "LLC", "Australasian", "Routledge", "Association", "pp", "In", "html"]
stop_words = text.ENGLISH_STOP_WORDS.union(stop_list)

# Declare the vectorizer. Note this is a TF-ID vectorizer, and thus weights low-frequency words,
# which is perfect for analyzing the very tribal lexicons used by philosophers.
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words= stop_words,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

# Extract the features for the training and test sets for Part 1.
features_train1 = tfidf.fit_transform(X_train1).toarray()
labels_train1 = y_train1
features_test1 = tfidf.transform(X_test1).toarray()
labels_test1 = y_test1

# Extract the features for the training and test sets for Part 2.
features_train2 = tfidf.fit_transform(X_train2).toarray()
labels_train2 = y_train2
features_test2 = tfidf.transform(X_test2).toarray()
labels_test2 = y_test2

#%% Plot the singular values

features_full1 = np.concatenate((features_train1,features_test1), axis=0)
features_full2 = np.concatenate((features_train2,features_test2), axis=0)
#%%
svd_plot(features_full2)


#%% Prints the most correlated uni/bi/trigrams

from sklearn.feature_selection import chi2

for field, index in sorted(label_index1.items()):
    features_chi2 = chi2(features_train1, labels_train1 == index)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    # trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
    print("# '{}' category:".format(field))
    print("  . ### Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . ### Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-5:])))
    # print("  . ### Most correlated trigrams:\n. {}".format('\n. '.join(trigrams[-5:])))
    print("")

#%% Classify with KNN, Part 1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(features_train1, labels_train1)
predict1 = neigh.predict(features_test1)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train1, neigh.predict(features_train1)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test1, predict1))

#%% Classify with KNN, Part 2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

neigh = KNeighborsClassifier(n_neighbors=9)
neigh.fit(features_train2, labels_train2)
predict2 = neigh.predict(features_test2)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train2, neigh.predict(features_train2)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test2, predict2))

#%% SVC Fit for comparison, Part1
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

svm_clf = svm.SVC()
svm_clf.fit(features_train1, labels_train1)
predict1 = svm_clf.predict(features_test1)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train1, svm_clf.predict(features_train1)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test1, predict1))

#%% SVC Fit for comparison, Part2
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

svm_clf = svm.SVC()
svm_clf.fit(features_train2, labels_train2)
predict2 = svm_clf.predict(features_test2)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train2, svm_clf.predict(features_train2)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test2, predict2))

#%% LDA for comparison Part 1
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


lda = LDA(n_components=1)
lda.fit(features_train1, labels_train1)
predict1 = lda.predict(features_test1)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train1, lda.predict(features_train1)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test1, predict1))

#%% LDA for comparison Part 2

lda = LDA(n_components=1)
lda.fit(features_train2, labels_train2)
predict2 = lda.predict(features_test2)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train2, lda.predict(features_train2)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test2, predict2))

#%% QDA for comparison Part 1
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

qda = QDA()
qda.fit(features_train1, labels_train1)
predict1 = qda.predict(features_test1)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train1, qda.predict(features_train1)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test1, predict1))

#%% QDA for comparison Part 2


qda = QDA()
qda.fit(features_train2, labels_train2)
predict2 = qda.predict(features_test2)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train2, qda.predict(features_train2)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test2, predict2))



#%% Visualize the features in a plot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

features = np.concatenate((features_train1,features_test1), axis=0)
labels = np.concatenate((labels_train1,labels_test1), axis=0)
title = "sigma = 2 PCA Components"
princ_comps = PCA(n_components=2).fit_transform(features)

# Put them into a dataframe````````````````````````````````````````````````````````
df_features = pd.DataFrame(data= princ_comps,
                           columns=['PC1', 'PC2'])

# Now we have to paste each row's label and its meaning
# Convert labels array to df
df_labels = pd.DataFrame(data=labels,
                         columns=['label'])

df_full = pd.concat([df_features, df_labels], axis=1)
df_full['label'] = df_full['label'].astype(str)

# Makes a new dictionary that is flipped, to unzip the label codes the other
# direction.
new_labels = {"0": "continental", "1": "analytic"}

# And map labels
df_full['label_name'] = df_full['label']
df_full = df_full.replace({'label_name': new_labels})

plt.figure(figsize=(10, 10))
sns.scatterplot(x='PC1',
                y='PC2',
                hue="label_name",
                data=df_full,
                palette=["red", "blue"],
                alpha=.7).set_title(title);

plt.savefig('part1_scatter.png', facecolor = "white")
plt.show()

#%% And for part 2

features = np.concatenate((features_train2,features_test2), axis=0)
labels = np.concatenate((labels_train2,labels_test2), axis=0)
title = "$\sigma$ = 2 PCA Components"
princ_comps = PCA(n_components=2).fit_transform(features)

# Put them into a dataframe
df_features = pd.DataFrame(data= princ_comps,
                           columns=['PC1', 'PC2'])

# Now we have to paste each row's label and its meaning
# Convert labels array to df
df_labels = pd.DataFrame(data=labels,
                         columns=['label'])

df_full = pd.concat([df_features, df_labels], axis=1)
df_full['label'] = df_full['label'].astype(str)

# Makes a new dictionary that is flipped, to unzip the label codes the other
# direction.
new_labels = {"0": "continental", "1": "analytic"}

# And map labels
df_full['label_name'] = df_full['label']
df_full = df_full.replace({'label_name': new_labels})

plt.figure(figsize=(10, 10))
sns.scatterplot(x='PC1',
                y='PC2',
                hue="label_name",
                data=df_full,
                palette=["red", "blue"],
                alpha=.7).set_title(title);

plt.savefig('part2_scatter.png', facecolor = "white")
plt.show()