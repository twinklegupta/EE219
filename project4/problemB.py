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


# The function removes some stemming words like go , going
# Also atakes care of punctuation marks, stop words etc
# Basically cleans the Data

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

# Loading the data of all caegories
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


for count in range(5):

    km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, precompute_distances='auto', verbose=0,
                copy_x=True, n_jobs=1, algorithm='auto').fit(twenty_tfidf_matrix)
    print(km.labels_.shape)

    # for i in km.labels_:
    #     print(i)
    #     print("#####")

    clustered_labels = []
    c = 0
    for i in labels:
        if (i < 4):
            clustered_labels.append(1)
        else:
            clustered_labels.append(0)
        c += 1
    clustered_labels1 = np.array(clustered_labels)
    print("Homogeneity: " % metrics.homogeneity_score(clustered_labels, km.labels_))
    print("Completeness: " % metrics.completeness_score(labels, km.labels_))
    print("V-measure: " % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: "
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: " % metrics.adjusted_mutual_info_score(clustered_labels, km.labels_))


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


#results :
# (2347, 42)
# (444, 1899)
# Error :  10.270498732

# (2354, 35)
# (473, 1870)
# Error :  10.7354184277

# (2350, 39)
# (470, 1873)
# Error :  10.7565511412

# (2348, 41)
# (448, 1895)
# Error :  10.3338968724

# (2361, 28)
# (548, 1795)
# Error :  12.1724429417


# Homogeneity: 0.539
# Completeness: 0.567
# V-measure: 0.276
# Adjusted Rand-Index: 0.145
# 1
# confusion matrix
# (2356, 33)
# (513, 1830)
# Error :  11.5384615385
# (4732,)


# Homogeneity: 0.549
# Completeness: 0.574
# V-measure: 0.280
# Adjusted Rand-Index: 0.148
# 2
# confusion matrix
# (2354, 35)
# (489, 1854)
# Error :  11.0735418428
# (4732,)


# Homogeneity: 0.637
# Completeness: 0.651
# V-measure: 0.323
# Adjusted Rand-Index: 0.179
# 3
# confusion matrix
# (2338, 51)
# (307, 2036)
# Error :  7.56551141167
# (4732,)


# Homogeneity: 0.563
# Completeness: 0.585
# V-measure: 0.287
# Adjusted Rand-Index: 0.154
# 4
# confusion matrix
# (2348, 41)
# (450, 1893)
# Error :  10.3761622992


# Homogeneity: 0.560
# Completeness: 0.582
# V-measure: 0.286
# Adjusted Rand-Index: 0.153
# 0
# confusion matrix
# (2349, 40)
# (458, 1885)
# Error :  10.5240912933
# (4732,)


# Homogeneity: 0.606
# Completeness: 0.624
# V-measure: 0.309
# Adjusted Rand-Index: 0.170
# 1
# confusion matrix
# (2337, 52)
# (355, 1988)
# Error :  8.60101437025
# (4732,)


# Homogeneity: 0.567
# Completeness: 0.589
# V-measure: 0.289
# Adjusted Rand-Index: 0.156
# 2
# confusion matrix
# (2347, 42)
# (441, 1902)
# Error :  10.2071005917
# (4732,)


# Homogeneity: 0.553
# Completeness: 0.577
# V-measure: 0.282
# Adjusted Rand-Index: 0.150
# 3
# confusion matrix
# (2354, 35)
# (482, 1861)
# Error :  10.9256128487
# (4732,)


# Homogeneity: 0.582
# Completeness: 0.603
# V-measure: 0.297
# Adjusted Rand-Index: 0.160
# 4
# confusion matrix
# (2349, 40)
# (419, 1924)
# Error :  9.69991546915


# Homogeneity: 0.542
# Completeness: 0.568
# V-measure: 0.278
# Adjusted Rand-Index: 0.148
# 0
# confusion matrix
# (2349, 40)
# (491, 1852)
# Error :  11.2214708369


################ final results ###################
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 0
# confusion matrix
# (2355, 34)
# (458, 1885)
# Error :  10.3972950127
# (4732,)
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 1
# confusion matrix
# (2351, 38)
# (470, 1873)
# Error :  10.7354184277
# (4732,)
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 2
# confusion matrix
# (2352, 37)
# (519, 1824)
# Error :  11.7497886729
# (4732,)
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 3
# confusion matrix
# (2346, 43)
# (430, 1913)
# Error :  9.99577345731
# (4732,)
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 4
# confusion matrix
# (2332, 57)
# (381, 1962)
# Error :  9.2561284869

# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 0
# confusion matrix
# (2331, 58)
# (374, 1969)
# Error :  9.12933220626
# (4732,)
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 1
# confusion matrix
# (2350, 39)
# (460, 1883)
# Error :  10.5452240068
# (4732,)
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 2
# confusion matrix
# (2353, 36)
# (482, 1861)
# Error :  10.9467455621
# (4732,)
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 3
# confusion matrix
# (2330, 59)
# (372, 1971)
# Error :  9.10819949281
# (4732,)
# Homogeneity:
# Completeness:
# V-measure:
# Adjusted Rand-Index:
# Adjusted mutual info score:
# 4
# confusion matrix
# (2352, 37)
# (479, 1864)
# Error :  10.9044801352