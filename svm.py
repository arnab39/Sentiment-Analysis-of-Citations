import os
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from scipy.sparse import hstack

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

corpus_file = os.path.join("test_set.txt")
with open(corpus_file, "r") as f:
	content = f.read().splitlines()

# create the dataframe
df = pd.DataFrame(columns=["sentence", "dependencies", "sentiment"])

for row in content:
	parse = re.match(r"([onp]),([\d]+),('.*'),(.*),('.*')",row)
	
	sentiment = parse.groups()[0]
	index = parse.groups()[1]
	sentence = parse.groups()[2]
	author = parse.groups()[3]
	dependencies = parse.groups()[4]
	
	if sentiment == "o":
		sentiment = 0
	elif sentiment == "p":
		sentiment = 1
	elif sentiment == "n":
		sentiment = 2

	sentence = sentence[1:-1]
	dependencies = dependencies[1:-1]
	
	df = df.append({"sentence":sentence, "dependencies":dependencies, "sentiment":sentiment}, ignore_index=True)

split = 726

# Original
# Tfidf vectors
s_vectorizer = TfidfVectorizer(ngram_range=(1,3))
s_X = s_vectorizer.fit_transform(df["sentence"][:-split].tolist())
print(s_X.shape)
d_vectorizer = TfidfVectorizer(ngram_range=(1,1))
d_X = d_vectorizer.fit_transform(df["dependencies"][:-split].tolist())
print(d_X.shape)
# append vectors
a_X = hstack([s_X, d_X])
# train the svm
svm_C = 1000
clf = SVC(C=svm_C, verbose=False, class_weight=None, kernel="rbf")
result = clf.fit(a_X, df["sentiment"][:-split].astype("int"))
# accuracy
test_s_X = s_vectorizer.transform(df["sentence"][-split:].tolist())
test_d_X = d_vectorizer.transform(df["dependencies"][-split:].tolist())
test_a_X = hstack([test_s_X, test_d_X])
print("Original Configuration:")
print("Macro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="macro"))
print("Micro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="micro"))
# print(confusion_matrix(df["sentiment"][-split:].astype("int"), result.predict(test_a_X)))

# Change to balanced loss
# Tfidf vectors
# s_vectorizer = TfidfVectorizer(ngram_range=(1,3))
# s_X = s_vectorizer.fit_transform(df["sentence"][:-split].tolist())
# print(s_X.shape)
# d_vectorizer = TfidfVectorizer(ngram_range=(1,1))
# d_X = d_vectorizer.fit_transform(df["dependencies"][:-split].tolist())
# print(d_X.shape)
# append vectors
# a_X = hstack([s_X, d_X])
# train the svm
svm_C = 1000
clf = SVC(C=svm_C, verbose=False, class_weight="balanced", kernel="rbf")
result = clf.fit(a_X, df["sentiment"][:-split].astype("int"))
# accuracy
test_s_X = s_vectorizer.transform(df["sentence"][-split:].tolist())
test_d_X = d_vectorizer.transform(df["dependencies"][-split:].tolist())
test_a_X = hstack([test_s_X, test_d_X])
print("\nBalanced loss (by inverse class frequency):")
print("Macro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="macro"))
print("Micro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="micro"))
# print(confusion_matrix(df["sentiment"][-split:].astype("int"), result.predict(test_a_X)))

# Change to smaller margin
# Tfidf vectors
# s_vectorizer = TfidfVectorizer(ngram_range=(1,3))
# s_X = s_vectorizer.fit_transform(df["sentence"][:-split].tolist())
# print(s_X.shape)
# d_vectorizer = TfidfVectorizer(ngram_range=(1,1))
# d_X = d_vectorizer.fit_transform(df["dependencies"][:-split].tolist())
# print(d_X.shape)
# append vectors
# a_X = hstack([s_X, d_X])
# train the svm
svm_C = 50000
clf = SVC(C=svm_C, verbose=False, class_weight="balanced", kernel="rbf")
result = clf.fit(a_X, df["sentiment"][:-split].astype("int"))
# accuracy
test_s_X = s_vectorizer.transform(df["sentence"][-split:].tolist())
test_d_X = d_vectorizer.transform(df["dependencies"][-split:].tolist())
test_a_X = hstack([test_s_X, test_d_X])
print("\nMake the margin smaller, C=", svm_C)
print("Macro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="macro"))
print("Micro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="micro"))
# print(confusion_matrix(df["sentiment"][-split:].astype("int"), result.predict(test_a_X)))

# Limit size of tfidf vectors
# Tfidf vectors
s_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=8192)
s_X = s_vectorizer.fit_transform(df["sentence"][:-split].tolist())
print("\n")
print(s_X.shape)
d_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=2048)
d_X = d_vectorizer.fit_transform(df["dependencies"][:-split].tolist())
print(d_X.shape)
# append vectors
a_X = hstack([s_X, d_X])
# train the svm
svm_C = 50000
clf = SVC(C=svm_C, verbose=False, class_weight="balanced", kernel="rbf")
result = clf.fit(a_X, df["sentiment"][:-split].astype("int"))
# accuracy
test_s_X = s_vectorizer.transform(df["sentence"][-split:].tolist())
test_d_X = d_vectorizer.transform(df["dependencies"][-split:].tolist())
test_a_X = hstack([test_s_X, test_d_X])
print("Limit size of TF-IDF vectors to 8192 for sentences and 2048 for dependencies:")
print("Macro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="macro"))
print("Micro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="micro"))
# print(confusion_matrix(df["sentiment"][-split:].astype("int"), result.predict(test_a_X)))

# Change to sentence bigrams
# Tfidf vectors
s_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=8192)
s_X = s_vectorizer.fit_transform(df["sentence"][:-split].tolist())
# print(s_X.shape)
d_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=2048)
d_X = d_vectorizer.fit_transform(df["dependencies"][:-split].tolist())
# print(d_X.shape)
# append vectors
a_X = hstack([s_X, d_X])
# train the svm
svm_C = 50000
clf = SVC(C=svm_C, verbose=False, class_weight="balanced", kernel="rbf")
result = clf.fit(a_X, df["sentiment"][:-split].astype("int"))
# accuracy
test_s_X = s_vectorizer.transform(df["sentence"][-split:].tolist())
test_d_X = d_vectorizer.transform(df["dependencies"][-split:].tolist())
test_a_X = hstack([test_s_X, test_d_X])
print("\nChange to bigrams for sentences:")
print("Macro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="macro"))
print("Micro F-score = ", f1_score(df["sentiment"][-split:].astype("int"), result.predict(test_a_X), average="micro"))
# print(confusion_matrix(df["sentiment"][-split:].astype("int"), result.predict(test_a_X)))