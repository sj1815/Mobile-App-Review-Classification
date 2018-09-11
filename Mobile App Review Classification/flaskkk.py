from flask import Flask, render_template, request
from evaluator import *
import json
import csv
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('file.html')

@app.route('/statistics', methods=['GET', 'POST'])
def MoveToAnotherPage():
    return render_template('file2.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify_this():
    if request.method == 'POST':
        review = request.form['review']
        result = classify(review)
        has_info_giving = int(result['has_info_giving'])
        has_info_asking = int(result['has_info_asking'])
        feature_request = int(result['feature_request'])
        bug_report = int(result['bug_report'])
        sentiment  = int(result['sentiment'])

        if has_info_giving == 1:
            has_info_giving = True
        elif has_info_giving == 0:
            has_info_giving = False

        if has_info_asking == 1:
            has_info_asking = True
        elif has_info_asking == 0:
            has_info_asking = False

        if feature_request == 1:
            feature_request = True
        elif feature_request == 0:
            feature_request = False

        if bug_report == 1:
            bug_report = True
        elif bug_report == 0:
            bug_report = False

        if sentiment == 1:
            sentiment = 'positive'
        elif sentiment == -1:
            sentiment = 'negative'
        elif sentiment == 0:
            sentiment = 'neutral'

        data = dict()
        data['Application feature details'] = has_info_giving
        data['User Queries'] = has_info_asking
        data['User Requests'] = feature_request
        data['Presence of bug'] = bug_report
        data['User Sentiment'] = sentiment

        return json.dumps(data)



if __name__ == "__main__":
    app.run(debug = True)