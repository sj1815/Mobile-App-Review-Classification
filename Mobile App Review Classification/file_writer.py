import csv
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
import nltk
nltk.download('stopwords')

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

        for r in data:
            reviews.append(r[7])
            has_info_giving.append(int(r[8]))
            has_info_asking.append(int(r[9]))
            feature_request.append(int(r[10]))
            bug_report.append(int(r[11]))
            sentiment.append(int(r[12]))

    reviews_has_info_giving = []
    for i in range(0,len(has_info_giving)):
        if int(has_info_giving[i]) == 1:
            reviews_has_info_giving.append(reviews[i])

    reviews_has_info_asking = []
    for i in range(0, len(has_info_asking)):
        if int(has_info_asking[i]) == 1:
            reviews_has_info_asking.append(reviews[i])

    reviews_has_feature_request = []
    for i in range(0, len(feature_request)):
        if int(feature_request[i]) == 1:
            reviews_has_feature_request.append(reviews[i])

    reviews_has_bug_report = []
    for i in range(0, len(bug_report)):
        if int(bug_report[i]) == 1:
            reviews_has_bug_report.append(reviews[i])

    # Stop word removal for info_giving
    final_stop_word_dict_info_giving = dict()
    stop_words = set(stopwords.words('english'))
    for i in range(len(reviews_has_info_giving)):
        word_tokens = word_tokenize(reviews_has_info_giving[i])
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        termList_has_info_giving = []
        for w in word_tokens:
            if w not in stop_words:
                termList_has_info_giving.append(w)

    for item in termList_has_info_giving:
        if item not in final_stop_word_dict_info_giving:
            final_stop_word_dict_info_giving[item] = 1
        else:
            final_stop_word_dict_info_giving[item] += 1

    #file = open("file_info_giving.txt","w")
    # Stop word removal for info_asking
    final_stop_word_dict_info_asking = dict()
    stop_words = set(stopwords.words('english'))
    for i in range(len(reviews_has_info_asking)):
        word_tokens = word_tokenize(reviews_has_info_asking[i])
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        termList_has_info_asking = []
        for w in word_tokens:
            if w not in stop_words:
                termList_has_info_asking.append(w)

    for item in termList_has_info_asking:
        if item  not in final_stop_word_dict_info_asking:
            final_stop_word_dict_info_asking[item] = 1
        else:
            final_stop_word_dict_info_asking[item] += 1

    # file = open("file_info_giving.txt","w")
# Stop word removal for feature_request
    final_stop_word_dict_feature_request = dict()
    stop_words = set(stopwords.words('english'))
    for i in range(len(reviews_has_feature_request)):
        word_tokens = word_tokenize(reviews_has_feature_request[i])
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        termList_has_feature_request = []
        for w in word_tokens:
            if w not in stop_words:
                termList_has_feature_request.append(w)

    for item in termList_has_feature_request:
        if item not in final_stop_word_dict_feature_request:
            final_stop_word_dict_feature_request[item] = 1
        else:
            final_stop_word_dict_feature_request[item] += 1

    # file = open("file_info_giving.txt","w")
    final_stop_word_dict_bug_report = dict()
    stop_words = set(stopwords.words('english'))
    for i in range(len(reviews_has_bug_report)):
        word_tokens = word_tokenize(reviews_has_bug_report[i])
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        termList_has_bug_report = []
        for w in word_tokens:
            if w not in stop_words:
                termList_has_bug_report.append(w)

    for item in termList_has_bug_report:
        if item not in final_stop_word_dict_bug_report:
            final_stop_word_dict_bug_report[item] = 1
        else:
            final_stop_word_dict_bug_report[item] += 1

    outliers = [',','.','!','?',':',';',')','(','I','Is','Im','p','But','...','If']
    dict_info_giving = dict()
    dict_info_asking = dict()
    dict_feature_request = dict()
    dict_bug_report = dict()
    for key in final_stop_word_dict_info_giving:
        if key not in outliers:
            value = final_stop_word_dict_info_giving[key]
            dict_info_giving[key] = value
    for key in final_stop_word_dict_info_asking:
        if key not in outliers:
            value = final_stop_word_dict_info_asking[key]
            dict_info_asking[key] = value
    for key in final_stop_word_dict_feature_request:
        if key not in outliers:
            value = final_stop_word_dict_feature_request[key]
            dict_feature_request[key] = value#final_stop_word_dict_feature_request[key]
    for key in final_stop_word_dict_bug_report:
        if key not in outliers:
            value = final_stop_word_dict_bug_report[key]
            dict_bug_report[key] = value

    file = open('file_info_giving','w')
    for key in dict_info_giving.keys():
        value = key + ':' + str(dict_info_giving[key])
        file.write(value+"\n")


    file = open('file_info_asking','w')
    for key in dict_info_asking.keys():
        value = key + ':' + str(dict_info_asking[key])
        file.write(value+"\n")

    file = open('file_feature_request', 'w')
    for key in dict_feature_request.keys():
        value = key + ':' + str(dict_feature_request[key])
        file.write(value + "\n")

    file = open('file_bug_report', 'w')
    for key in dict_bug_report.keys():
        value = key + ':' + str(dict_bug_report[key])
        file.write(value + "\n")

    print()
def main():
   read_csv("reviews.csv")

if __name__ == '__main__':
    main()