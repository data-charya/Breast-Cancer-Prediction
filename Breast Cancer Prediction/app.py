# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


df = pd.read_csv("cancer.csv")


st.write("""
	# **Breast Cancer Prediction :heart: **
	""")
st.subheader("- Shanwill Pinto")

X = df.loc[:,["radius_mean", "area_mean","radius_worst", "compactness_worst", "symmetry_worst"]]
y = df.diagnosis

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)

y_predict = model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


st.sidebar.header("User Input")

def user_input_features():
	radius_mean = st.sidebar.slider('Radius Mean',0.0,40.0,8.0)
	area_mean = st.sidebar.slider('Area Mean',100.0,1500.0,150.0)
	radius_worst = st.sidebar.slider('Radius Worst',0.0,40.0,8.0)
	compactness_worst = st.sidebar.slider('Compactness Worst',0.0,40.0,8.0)
	symmetry_worst = st.sidebar.slider('Symmetry Worst',0.0,40.0,8.0)
	data = {'radius_mean': radius_mean,'area_mean': area_mean,'radius_worst': radius_worst,'compactness_worst': compactness_worst,'symmetry_worst':symmetry_worst}
	features = pd.DataFrame(data, index =[0])
	return features

dataf = user_input_features()
 
st.subheader(""" User Inputs :inbox_tray: """)
st.write(dataf)

Prediction = model.predict(dataf)


st.write(""" # Prediction """)
if Prediction == 'M':
	st.write(""" ### Malignant :red_circle: """)
elif Prediction == 'B':
	st.write(""" ### Benign :white_check_mark: """)



st.write("""
	# Model Accuracy
	""")
st.write(accuracy_score(y_test,y_predict))