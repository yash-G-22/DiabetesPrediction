from flask import Flask,render_template,request,app,jsonify,Response
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
scaler = pickle.load(open('D:/Diabetes_Prediction/model/standardscaler.pkl','rb'))
model = pickle.load(open('D:/Diabetes_Prediction/model/logistic_regression.pkl','rb'))

@app.route('/')
def index():
    return render_template('home.html')
@app.route('/predictdata',methods=['POST'])
def predict_datapoint():
    result = ""
    Pregnancies = int(request.form.get('Pregnancies'))
    Glucose = float(request.form.get("Glucose"))
    BloodPressure = float(request.form.get("BloodPressure"))
    SkinThickness = float(request.form.get("SkinThickness"))
    Insulin = float(request.form.get("Insulin"))
    BMI = float(request.form.get("BMI"))
    DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
    Age = float(request.form.get("Age"))

    new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    predicted = model.predict(new_data)
    if predicted[0] == 1:
        result = "DIABETIC"
    else:
        result = "NON-DIABETIC"

    return render_template("home.html",result = result)
    
if __name__ =="__main__":
    app.run(host="0.0.0.0")