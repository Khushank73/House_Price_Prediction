import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv('USA_Housing.csv')
X=dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=dataset['Price']
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
# print(model.predict([[4,5,6,7,78]]))

