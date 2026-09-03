import streamlit as st

st.title("Ask My PDF Bot")

hf_token = st.sidebar.text_input(
    "Enter Hugging Face API Token",
    type="password"
)

if not hf_token:
    st.sidebar.info("Enter your Hugging Face API token.")
    st.stop()

st.success("Python code is working correctly!")
