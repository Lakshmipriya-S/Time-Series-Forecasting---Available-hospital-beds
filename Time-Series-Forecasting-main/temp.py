from flask import Flask,render_template,request,jsonify
import pickle

#from flask_cors import CORS, cross_origin
import numpy as np

temp=Flask(__name__,template_folder='template')
#CORS(temp)
model=pickle.load(open('prophet.pkl','rb'))

@temp.route('/')
def home():
    return render_template("home.html")

@temp.route('/about')
def about():
    return render_template('about.html')

@temp.route('/contact')
def contact():
    return render_template('contact.html')

@temp.route('/predict',methods = ['POST'])
def predict():
    
    int_features = int(request.form['x'])
    future = model.make_future_dataframe(periods=int_features)
    forecast = model.predict(future)
    o=forecast[['ds','yhat']].tail(int_features)
    o['ds']=o['ds'].astype(object)
    dates=[]
    beds=[]
    for x in range(366,o.index[-1]):
        r=str(o.ds[x])[:10]
        dates.append(r)
        s=int(o.yhat[x])
        beds.append(s)
    

    
    
    return render_template("predict.html",l=int_features,dates=dates,beds=beds)

if __name__ =="__main__":
    temp.run(debug =True)