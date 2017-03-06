"""Script to download the 20 newsgroups text classification set"""

import os
import tarfile
from urllib2 import urlopen

url = ("http://people.csail.mit.edu/jrennie/"
       "20Newsgroups/20news-bydate.tar.gz")

folder_name = url.rsplit('/', 1)[1]
training_f = "20news-bydate-train"
test_f = "20news-bydate-test"

print("opening url nd getting all data and opening folder")
opener = urlopen(url)
open(folder_name, 'wb').write(opener.read())

tarfile.open(folder_name, "r:gz").extractall(path='.')
os.remove(folder_name)