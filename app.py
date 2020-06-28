from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('iris.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():

    int_features = [x for x in request.form.values()]
    arr=[np.array(int_features)]
    #arr=np.array([[data1,data2,data3,data4]])
    pred=model.predict(arr)

    return render_template('after.html',data=pred)

if __name__ == "__main__":
    app.run(debug=True)