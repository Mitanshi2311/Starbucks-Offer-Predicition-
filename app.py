import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Random_Regressor_model.sav', 'rb'))
model2 = pickle.load(open('Random_Regressor_model2.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/offer-successful')
def page():
    return render_template('page1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    label = {1:"discount", 2:"bogo",3:"informational"}
    return render_template('index.html', prediction_text='Recommended offer is {}'.format(label[output]))

@app.route('/predict2',methods=['POST'])
def predict2():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model2.predict(final_features)

    output = round(prediction[0], 2)
    label = {1:"successful", 0:"not successful"}
    return render_template('page1.html', prediction_text='Recommended offer will {}'.format(label[output]))


if __name__ == "__main__":
    app.run(debug=True)