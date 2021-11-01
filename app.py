
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction=='0':
        print('True')
    else:
        print('False')

    output = round(prediction[0], 2)
    if output==0:
        go="Negative"
    else:
        go="Positive"

    return render_template('index.html', prediction_text='The Diagnosis result is {}'.format(go))


if __name__ == "__main__":
    app.run(debug=True)