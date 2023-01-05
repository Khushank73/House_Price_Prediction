import numpy as np
from flask import Flask,render_template,request,jsonify
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int (x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    predictions=model.predict(final_features)
    output=round(predictions[0],2)
    return render_template('index.html',prediction_text='Estimated House Price is $ {}' .format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]
    return jsonify(output)



if __name__=='__main__':
    app.run(debug=True)

