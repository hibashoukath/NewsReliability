# This is a sample Python script.
import re
import pickle as pk
import gensim
from flask import Flask, request, render_template
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])

def reliablity():
    input_pred = None
    result = None
    if request.method == "POST":

        text1 = request.form.get("text")
        print(text1)
        input_text = text1
        input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)

        filename = r'svm_model.pkl'
        model = gensim.models.Word2Vec.load('word2vec_model.bin')

        loaded_model = joblib.load(filename)
        input_tokens = input_text.lower().split()

        input_vector = np.zeros((1, model.vector_size))
        for token in input_tokens:
            if token in model.wv:
                input_vector += model.wv[token]
        input_vector /= len(input_tokens)

        input_pred = loaded_model.predict(input_vector)
        input_pred = input_pred.astype(int)

#news reliability is predicted

    if input_pred == 1:
        print("Review is Positive")
        result = 'Review is Positive'
    else:
        print("Review is Negative")
        result = "Review is negative"

    return render_template('index.html', prediction_text='News Reliability Anlaysis: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
