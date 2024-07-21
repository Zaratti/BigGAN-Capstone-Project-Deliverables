import pickle 
import streamlit as st
import numpy as np
import io


import tensorflow as tf

from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_models():
  Elec_model = load_model('Elec_model.h5')
  Elec_model._make_predict_function()
  Elec_model.summary()
  return Elec_model
  

## scale_file = joblib.load('scaler.pkl')
## return scale_file

def show_predict_page():
 st.title("Electricity Prediction Model")

 st.write("""### We need some information to predict""")