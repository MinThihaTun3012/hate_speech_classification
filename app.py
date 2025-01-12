import streamlit as st 
import tensorflow as tf
import pickle as pk
import simmpst
from simmpst.tokenization import MultilingualPartialSyllableTokenization
from utils import prepare_text, get_prediction, get_random_text

# Load the model and tokenizer
try:
    model = tf.keras.models.load_model('model_best_weights.keras')
    with open("partial_syllable.model.pkl", "rb") as f:
        tokenizer = pk.load(f)
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Hate Speech Classification",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.snow()

# Initialize session state for user_input
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

#<----------------app start ---------------------->


st.title("Hate Speech Classification")

threshold = st.slider("Threshold ", 0.0, 1.0, value=0.5)

# Text area for user input
user_input = st.text_area("Enter a Text: 300 char Max and Only Burmese language", 
                          max_chars=300, 
                          value=st.session_state.user_input)

# Set button type based on whether user_input has text
if user_input:
    button_type = "primary"  
else:
    button_type = "secondary"  

col1, col2 = st.columns([5.8 , 1.5], gap="large")

with col1: 
    if st.button("Make Prediction", type=button_type):
        with st.spinner("Predicting..."):
            try:
                tokens = prepare_text(user_input, tokenizer)
                prediction = get_prediction(tokens, threshold, model)
                st.success(f"Model Predicts: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

with col2:
    if st.button("Get test data"):
        st.session_state.user_input = get_random_text()
        st.info("Test data loaded into text area.")

#<-------------- app end ------------------------>

