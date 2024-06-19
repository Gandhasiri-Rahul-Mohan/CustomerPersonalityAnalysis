import streamlit as st 
import pandas as pd 
import numpy as np

from sklearn.svm import SVC
import pickle 

from sklearn.preprocessing import StandardScaler 
 



# Function to predict using the saved model
def cluster(age, education, marital_status, income, kids):

    features = [[age, education, marital_status, income, kids]]
    model = pickle.load(open('SVM.pkl', 'rb'))
    prediction = model.predict(features)
    return prediction

st.set_page_config(page_title="Clustering")
st.title('Customer Clustering')
st.write('This app retuns a cluster id to which a customer may belong : ')

# Add input widgets for user input
col1, col2 = st.columns(2)
age = col1.slider('Age', min_value=18, max_value=100, value=30, step=1)

education = col2.radio('Education', ['UnderGraduate', 'Graduate', 'PostGraduate', 'PhD'])

marital_status = col2.radio( "Marital Status", ('Single', 'Married', 'Together', 'Others') )

income = col1.slider('Income', min_value=0, max_value=600000, value=50000, step=1000)

kids = int(col2.radio( "Select Number Of Kids In Household", ('0', '1','2','3') ))

# Map marital status and education to numerical values
marital_status_map = {'Single':0,'Married':1,'Together':2,'Others':3}
education_map = {'UnderGraduate':0,'Graduate':1, 'PostGraduate':2, 'PhD': 3}

marital_status = marital_status_map[marital_status]
education = education_map[education]


st.write(f'Your inputs were : {age, education, marital_status, income, kids}')

# Make a prediction and display the result
if st.button('Predict'):
    cluster_id = cluster(age, education, marital_status, income, kids)
    st.write('**Predicted Cluster id :**', cluster_id)