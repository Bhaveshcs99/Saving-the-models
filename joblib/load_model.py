import joblib

# load the model
loaded_model = joblib.load('dib_79.pkl')
pred = loaded_model.predict([[10,20,30,40,50,10,20,10]])
if pred[0]==1:
    print('Person is diabetic')
else:
    print('Person is not diabetic')