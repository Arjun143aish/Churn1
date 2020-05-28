import numpy as np
import pandas as pd
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
    feature_name = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard',
                    'IsActiveMember','EstimatedSalary','Geography_Germany',
                    'Geography_Spain','Gender_Male']
    
    df = pd.DataFrame(final_features, columns = feature_name)
    prediction = model.predict(df)

    output = prediction
    
    if output > 0.5:
        status = 'Customer will Churn'
    else:
        status ='Customer will not Churn'

    return render_template('index.html', prediction_text='Status of Application: {}'.format(status))


if __name__ == "__main__":
    app.run(debug=True)