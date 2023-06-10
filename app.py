import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 
import sklearn

app=Flask(__name__)

model=pickle.load(open('randomforest1.pkl','rb'))
@app.route('/')
def home():
    #return hello world
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    #data is dictionary name we  use to store our input in postman
    print(data)
    output=model.predict([list(data.values())])[0] 
    return jsonify(output)


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_feature=[np.array(data)]
    #data is dictionary name we  use to store our input in postman
    output=model.predict(final_feature)[0]
    print(output) 
    return render_template('home.html',prediction_text="Airfoil pressure is {}".format(output))


if  __name__=='__main__':
    app.run(debug=True)

