import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')


#load the boston house price dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=['MEDV'])

#sidebar
# Header of Specify Input Parameters
st.sidebar.title("Specify Input Parameters")

def user_input_features():
    st.sidebar.write("""for bla bla bla""")
    CRIM = st.sidebar.slider('CRIM', X['CRIM'].min(), X['CRIM'].max(), X['CRIM'].mean())
    ZN = st.sidebar.slider('ZN', X['ZN'].min(), X['ZN'].max(), X['ZN'].mean())
    INDUS = st.sidebar.slider('INDUS', X['INDUS'].min(), X['INDUS'].max(), X['INDUS'].mean())
    CHAS = st.sidebar.slider('CHAS', X['CHAS'].min(), X['CHAS'].max(), X['CHAS'].mean())
    NOX = st.sidebar.slider('NOX', X['NOX'].min(), X['NOX'].max(), X['NOX'].mean())
    RM = st.sidebar.slider('RM', X['RM'].min(), X['RM'].max(), X['RM'].mean())
    AGE = st.sidebar.slider('AGE', X['AGE'].min(), X['AGE'].max(), X['AGE'].mean())
    DIS = st.sidebar.slider('DIS', X['DIS'].min(), X['DIS'].max(), X['DIS'].mean())
    RAD = st.sidebar.slider('RAD', X['RAD'].min(), X['RAD'].max(), X['RAD'].mean())
    TAX = st.sidebar.slider('TAX', X['TAX'].min(), X['TAX'].max(), X['TAX'].mean())
    PTRATIO = st.sidebar.slider('PTRATIO', X['PTRATIO'].min(), X['PTRATIO'].max(), X['PTRATIO'].mean())
    B = st.sidebar.slider('B', X['B'].min(), X['B'].max(), X['B'].mean())
    LSTAT = st.sidebar.slider('LSTAT', X['LSTAT'].min(), X['LSTAT'].max(), X['LSTAT'].mean())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0]) 
    return features

df = user_input_features()

#main panel

#Print specified input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

#Model Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
#Apply model to the user input features
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

#Explaining model prediction using SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')


plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type='bar')
st.pyplot( box_inches='tight')
st.write('---')