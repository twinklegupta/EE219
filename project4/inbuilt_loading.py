from sklearn.datasets import fetch_20newsgroups


categories = [ 'comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'rec.autos', 'rec.motorcycles','rec.sport.baseball']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=40)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=40)

print()