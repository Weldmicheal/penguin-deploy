import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Visualization and Prediction App

This app predicts the **Palmer Penguin** species!

""")
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type = ["CSV"])
#st.set_option('deprecation.showfileUploaderEncoding', False)

if uploaded_file is not None:
	input_df = pd.read_csv(uploaded_file)
else:
	def user_input_features():
		island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
		sex = st.sidebar.selectbox('Sex', ('male', 'female'))
		bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
		bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
		flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
		body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
		data = {'island': island,
			'bill_length_mm': bill_length_mm,
			'bill_depth_mm': bill_depth_mm,
			'flipper_length_mm': flipper_length_mm,
			'body_mass_g': body_mass_g,
			'sex': sex}
		features = pd.DataFrame(data, index = [0])
		return features
	input_df = user_input_features()

@st.cache
def read_penguins():
		penguins_r = pd.read_csv('penguins_cleaned.csv')
		return penguins_r
penguins_raw = read_penguins()



st.subheader('Penguin dataset')
st.write(penguins_raw)

penguins_sp = penguins_raw[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
penguins_gr = penguins_sp.groupby(['species']).mean()
st.subheader('Penguin by species')
st.write(penguins_gr)

penguins_gr.plot(kind = 'bar', figsize = (10, 6))
st.pyplot()
	
st.subheader('Customizable plot')

plot_choice = st.selectbox('select type of plot', ('bar','pie'))

if plot_choice == 'pie':
	penguin_par = st.selectbox('select penguin part to plot', ('bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g'))

	if st.button('Generate Plot'):
		st.write('Generating plot of {}'.format(penguin_par))
		
		penguins_gr[penguin_par].plot( kind = 'pie', labels = None,  autopct="%1.1f%%", pctdistance=1.12, explode = [0, 0.1, 0.2], figsize=(10, 6),)

		plt.legend(labels=penguins_gr.index, loc='upper left')
		plt.axis('equal')
		st.pyplot()
elif plot_choice:
	penguin_par = st.selectbox('select penguin part to plot', ('bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g'))

	if st.button('Generate Plot'):
		st.write('Generating plot of {}'.format(penguin_par))
		penguins_gr[penguin_par].plot(kind = 'bar', figsize = (6, 6))
		st.pyplot()
		

penguins = penguins_raw.drop(columns = ['species'])
df = pd.concat([input_df, penguins], axis = 0)

encode = ['sex', 'island']
for col in encode:
	dummy = pd.get_dummies(df[col], prefix = col)
	df = pd.concat([df, dummy], axis = 1)
	del df[col]
df = df[:1]
		
st.subheader('User Input features')

if uploaded_file is not None:
	st.write(df)
else:
	st.write('Awaiting CSV file to be uploaded.')
	st.write(df)
	
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)