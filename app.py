import numpy as np
from flask import Flask,render_template,request
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int (x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    predictions=model.predict(final_features)
    output=round(predictions[0],2)
    return render_template('index1.html',prediction_text='Estimated House Price is $ {}' .format(output))


if __name__=='__main__':
    app.run(debug=True)