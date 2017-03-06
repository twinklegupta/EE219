import re

from nltk.corpus import stopwords
import nltk.stem
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import numpy as np
import scipy
from sklearn.decomposition import TruncatedSVD

stemmer = nltk.stem.SnowballStemmer('english')
stop_words = text.ENGLISH_STOP_WORDS  # stopwords


def clean_data(temp):

    temp = temp.lower()

    # This statement will remove all the punctuation marks
    tokenize1 = RegexpTokenizer(r'\w+')

    # Stemmer is used to stem the words, eg. -ing removal
    stemming = PorterStemmer()

    # The tokenize function creates tokens from the text
    token = tokenize1.tokenize(temp)

    # This statement removes all the stop words such as articles that do not contribute towards the meaning of the text
    no_stops = [x for x in token if not x in stop_words]

    # This statement will perform the actual stemming
    no_ing = [stemming.stem(y) for y in no_stops]
    results = [stemmer.stem(y) for y in no_ing]

    # This statement removes all the digits
    results = [z for z in results if not z.isdigit()]

    # Each word is appended with a space before it
    return " ".join(results)





categories = [
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey'
]

# Loading the data of all categories
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,
                                  remove=('headers', 'footers'))

labels = twenty_train.target

print("labels :- ", labels.shape)

# Get length of training data
len_twenty_data, = twenty_train.filenames.shape

# Passing every document for processing.
count = 0
while count < len_twenty_data:
    temp = twenty_train.data[count]
    twenty_train.data[count] = clean_data(temp)
    count += 1

# create the TFxIDF vector representations for all docs

count_vect = CountVectorizer()
twenty_td_vector = count_vect.fit_transform(twenty_train.data)

# Calculate the required txidf matrix by transorming the above formed vector and get categoriesteh required dimensions
tf_transformer = TfidfTransformer(use_idf=True).fit(twenty_td_vector)
twenty_tfidf_matrix = tf_transformer.transform(twenty_td_vector)
doc_count, term_count = twenty_tfidf_matrix.shape
# print(twenty_tfidf_matrix)

print ("Total number of terms are", term_count)



print(twenty_td_vector.shape)
U, s, V = scipy.sparse.linalg.svds(twenty_tfidf_matrix)
print(U.shape)
print(s.shape)
print(V.shape)
print(s)
dimension, = s.shape
svd = TruncatedSVD(n_components=dimension,n_iter=10)
svd.fit(twenty_tfidf_matrix)
twenty_tfidf_matrix_r=svd.fit_transform(twenty_tfidf_matrix)
print(twenty_tfidf_matrix_r)


# varying dimensions to find variances and best dimension ## as Sanity check
for dim in range(2,100):
    svd = TruncatedSVD(n_components=dim, n_iter=10)
    svd.fit(twenty_tfidf_matrix)
    twenty_tfidf_matrix_r = svd.fit_transform(twenty_tfidf_matrix)
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, precompute_distances='auto', verbose=0,
                copy_x=True, n_jobs=1, algorithm='auto').fit(twenty_tfidf_matrix_r)
    clustered_labels = []
    c = 0
    for i in labels:
        if (i < 4):
            clustered_labels.append(1)
        else:
            clustered_labels.append(0)
            c += 1
    clustered_labels1 = np.array(clustered_labels)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(clustered_labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(labels, km.labels_))


    c = 0;
    comCorrect = 0
    comIncorrect = 0
    recCorrect = 0
    recIncorrect = 0

    for i in km.labels_:
        if(i == 0 and clustered_labels1[c] == 0): recCorrect += 1
        elif (i == 1 and clustered_labels1[c] == 0): recIncorrect += 1
        elif (i == 1 and clustered_labels1[c] == 1): comCorrect += 1
        else : comIncorrect += 1
        c += 1


    if(comCorrect + recCorrect < comIncorrect + recIncorrect):
        comCorrect,comIncorrect = comIncorrect,comCorrect
        recCorrect,recIncorrect = recIncorrect,recCorrect


    # print("comCorrect : ",comCorrect)
    # print("comIncorrect : ",comIncorrect)
    # print("recCorrect : ",recCorrect)
    # print("recIncorrect : ",recIncorrect)
    print(count)
    print("confusion matrix")

    print(recCorrect, recIncorrect)
    print(comIncorrect, comCorrect)

    print "Error : ",(float(comIncorrect+recIncorrect)/float(comIncorrect+comCorrect+recIncorrect+recCorrect))*100


# # for count in range(5):
#
# km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, precompute_distances='auto', verbose=0,
#                 copy_x=True, n_jobs=1, algorithm='auto').fit(twenty_tfidf_matrix_r)
# print(km.labels_.shape)
#
# clustered_labels = []
# c = 0
# for i in labels:
#     if (i < 4):
#         clustered_labels.append(1)
#     else:
#         clustered_labels.append(0)
#         c += 1
# clustered_labels1 = np.array(clustered_labels)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(clustered_labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(labels, km.labels_))
#
#
# c = 0;
# comCorrect = 0
# comIncorrect = 0
# recCorrect = 0
# recIncorrect = 0
#
# for i in km.labels_:
#     if(i == 0 and clustered_labels1[c] == 0): recCorrect += 1
#     elif (i == 1 and clustered_labels1[c] == 0): recIncorrect += 1
#     elif (i == 1 and clustered_labels1[c] == 1): comCorrect += 1
#     else : comIncorrect += 1
#     c += 1
#
#
# if(comCorrect + recCorrect < comIncorrect + recIncorrect):
#     comCorrect,comIncorrect = comIncorrect,comCorrect
#     recCorrect,recIncorrect = recIncorrect,recCorrect
#
#
# # print("comCorrect : ",comCorrect)
# # print("comIncorrect : ",comIncorrect)
# # print("recCorrect : ",recCorrect)
# # print("recIncorrect : ",recIncorrect)
# print(count)
# print("confusion matrix")
#
# print(recCorrect, recIncorrect)
# print(comIncorrect, comCorrect)
#
# print "Error : ",(float(comIncorrect+recIncorrect)/float(comIncorrect+comCorrect+recIncorrect+recCorrect))*100
