import csv
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

def read_csv(filename):
    with open(filename, 'rt', encoding="utf8") as csvfile:
        filereader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(filereader)
        data = []
        reviews = []
        has_info_asking = []
        has_info_giving = []
        feature_request = []
        bug_report = []
        sentiment = []
        for r in filereader:
            if(len(r[7]) > 0):
                data.append(r)
        shuffle(data)
        for r in data:
            reviews.append(r[7])
            has_info_giving.append(int(r[8]))
            has_info_asking.append(int(r[9]))
            feature_request.append(int(r[10]))
            bug_report.append(int(r[11]))
            sentiment.append(int(r[12]))
        return[data, reviews, has_info_giving, has_info_asking,
               feature_request, bug_report, sentiment]

def build_classifier(reviews, labels, train_test_break, name):

    X_train = reviews[0:train_test_break+1]
    Y_train = labels[0:train_test_break+1]

    sm = SMOTE()

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    c = 0
    c1 = 0
    for i in Y_train:
        if i == 1:
            c1 += 1
        if i == 0:
            c += 1
    print("Before resampling")
    print(str(c) + " " + str(c1))

    X_train_res, Y_train_res = sm.fit_sample(X_train_counts, Y_train)

    c = 0
    c1 = 0
    for i in Y_train_res:
        if i == 1:
            c1 += 1
        if i == 0:
            c += 1
    print("After resampling")
    print(str(c) + " " + str(c1))

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_res)

    clf = MultinomialNB().fit(X_train_tfidf, Y_train_res)
    joblib.dump(clf, 'classifiers/'+name+'_classifier.sav')
    joblib.dump(count_vect, 'counts/'+name+'_counts.sav')
    joblib.dump(tfidf_transformer, 'tfidf/'+name+'_tfidf.sav')

def test_classifier(reviews, class_labels,
                                    train_test_break, name):
    X_test = reviews[train_test_break+1:]
    Y_test = class_labels[train_test_break + 1:]

    count_vect = joblib.load('counts/'+name+'_counts.sav')
    X_test_counts = count_vect.transform(X_test)

    tfidf_transformer = joblib.load('tfidf/'+name+'_tfidf.sav')
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    clf = joblib.load('classifiers/'+name+'_classifier.sav')
    predicted = clf.predict(X_test_tfidf)
    c = 0
    c1 = 0
    for i in predicted:
        if i == 1:
            c1+=1
        if i == 0:
            c+=1
    print(str(c) + " "+ str(c1))
    print(np.mean(Y_test == predicted))

def stem_lemmatize_tokens(tokens, lemmatizer, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(lemmatizer.lemmatize(stemmer.stem(item)))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stems = stem_lemmatize_tokens(tokens, lemmatizer, stemmer)
    return stems

def main():
    data, reviews, has_info_giving, has_info_asking, feature_request, \
    bug_report, sentiment = read_csv("reviews.csv")
    train_test_break = round(len(reviews) * 0.7)
    classes = [has_info_giving, has_info_asking, feature_request, bug_report,
               sentiment]
    names = ["has_info_giving", "has_info_asking", "feature_request",
             "bug_report", "sentiment"]
    for c in range(len(classes)):
        print("\nBuilding "+names[c]+" classifier")
        build_classifier(reviews, classes[c],
                                         train_test_break, names[c])
        print("Testing " + names[c] + " classifier")
        test_classifier(reviews, classes[c],
                                        train_test_break, names[c])
if __name__ == '__main__':
    main()