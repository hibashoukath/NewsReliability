from flask import Flask, request, render_template
import pickle as pk
import re
import gensim
import numpy as np
# from sklearn.ensemble import RandomForestRegressor

import pandas as pd

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods=["GET", "POST"])
#reliablity function
#this function gets the request parameters
#finalized model is saved as a pickle file and accessed
# request parameters are converted to appropriate values and passed to the model
def reliablity():
    if request.method == "POST":

        text1 = request.form.get("text")
        input_text = text1
        input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)

        filename = r'svm_model.pk'
        model = gensim.models.Word2Vec.load('word2vec_model.bin')
        loaded_model = pk.load(open(filename, 'rb'))
        input_tokens = input_text.lower().split()


        input_vector = np.zeros((1, model.vector_size))
        for token in input_tokens:
            if token in model:
                input_vector += model[token]
                input_vector /= len(input_tokens)
        input_pred = loaded_model.predict(input_vector)
        input_pred = input_pred.astype(int)


#news reliability is predicted


    if input_pred[0] == 1:
        print("Review is Positive")
        result = 'Review is Positive'
    else:
        print("Review is Negative")
        result = "Review is negative"
    return render_template('index.html', prediction_text='News Reliability Anlaysis: {}'.format(result))


if __name__ == '__main__':
    app.run(debug=True)
