import streamlit as st
import joblib
# Load model
model = joblib.load('model/iris_model.pkl')
st.title('Iris Flower Classifier')
# Input features
sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length', 1.0, 7.0, 1.5)
petal_width = st.slider('Petal Width', 0.1, 2.5, 0.2)
# Predict
if st.button('Predict'):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    species = ['setosa', 'versicolor', 'virginica']
    st.success(f'Predicted: {species[prediction[0]]}')