import pandas as pd
import pickle
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

'''
text_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2, 
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
'''
# Load Train Model
model = tf.keras.models.load_model('./mode.h5')

# Load Pickel Files
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('onehot_encode_geo.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)




print("hello", geo_encoder.categories_, label_encoder.classes_)


# StreamLit App
st.title('Customer churn Prediction')  

#User Input
geography = st.selectbox('Select Country', geo_encoder.categories_[0])
gender = st.selectbox('Select Gender', label_encoder.classes_)
age = st.slider('Select Age', 22, 90)
creditScore = st.number_input('Enter Credit Score')
balance = st.number_input('Enter Balance')
estimated_salary = st.number_input('Enter Salary')
tenure = st.slider('Select Tenure', 0, 10)
number_of_products = st.slider('Select Num Of Products', 0, 4)
has_cr_card = st.selectbox('Select Has Credit Card', [0,1])
is_active_member = st.selectbox('Select Is Active Member', [0,1])

#Input Data
input_data_pf=  pd.DataFrame({
    'CreditScore': [creditScore],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products], 
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data_pf.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

prediction_pro = prediction[0][0]

st.write(f'Churn Probability: {prediction_pro:.2f}')   

if(prediction_pro > 0.5):
    st.write('The customer is likely to churn.')
else:
    st.write('The Customer is not likely to churn.')    





