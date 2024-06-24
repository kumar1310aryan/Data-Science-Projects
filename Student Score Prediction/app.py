import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

#importing our linear regression model
linear_model=pickle.load(open('models/linear.pkl','rb'))



## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictscore',methods=['GET','POST'])
def predict_score():
    if request.method=='POST':
        Hours=float(request.form.get('Hours'))
        
        result=linear_model.predict(np.array([[Hours]]))

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')




if __name__=="__main__":
    app.run(host="0.0.0.0")