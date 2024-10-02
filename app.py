import streamlit as st
from transformers import pipeline

# Load the pre-trained question-answering pipeline
question_answering = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Streamlit app layout
st.title("Question Answering with BERT")

st.write("""
### Enter the passage and your question, and BERT will try to answer it!
""")

# Text input for passage and question
passage = st.text_area("Passage", "Enter a passage here...")
question = st.text_input("Question", "Enter your question...")

if st.button("Get Answer"):
    if passage and question:
        # Perform question answering
        result = question_answering(question=question, context=passage)
        
        # Display the result and confidence score
        st.write(f"**Answer**: {result['answer']}")
        st.write(f"**Confidence Score**: {result['score']:.2f}")
    else:
        st.write("Please provide both a passage and a question.")

