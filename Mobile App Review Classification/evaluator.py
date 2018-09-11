import sys
from sklearn.externals import joblib

def classify(review):
    X_test = [review]
    names = ["has_info_giving", "has_info_asking", "feature_request",
             "bug_report", "sentiment"]
    predictions = {}
    for name in names:
        count_vect = joblib.load('counts/' + name + '_counts.sav')
        X_test_counts = count_vect.transform(X_test)

        tfidf_transformer = joblib.load('tfidf/' + name + '_tfidf.sav')
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)

        clf = joblib.load('classifiers/' + name + '_classifier.sav')
        predicted = clf.predict(X_test_tfidf)
        predictions[name] = predicted[0]
    return predictions


if __name__ == '__main__':
    review = sys.argv[1]
    print("Input is: " + review)
    print("Predictions are: ")
    print(classify(review))
