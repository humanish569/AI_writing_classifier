import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import streamlit as st

# Load the model
def load_model(model_path):
    st.write("Loading model...")
    model = tf.keras.models.load_model(model_path)
    return model

# Load the tokenizer
def load_tokenizer(tokenizer_path):
    st.write("Loading tokenizer...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Preprocess the input text
def preprocess_text(text, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

# Evaluate the text
def evaluate_text(text, model, tokenizer, max_len=100):
    # Preprocess the text
    padded_sequence = preprocess_text(text, tokenizer, max_len)
    
    # Make a prediction
    prediction = model.predict(padded_sequence)
    
    # Get the predicted class and confidence scores
    class_labels = ['Human Written', 'AI Written', 'AI Written (Humanized)']
    predicted_class = class_labels[np.argmax(prediction)]
    confidence_scores = {label: float(score) for label, score in zip(class_labels, prediction[0])}
    
    return predicted_class, confidence_scores

# Streamlit UI
def main():
    st.title("Text Classification UI")
    st.write("Enter text to classify it as Human Written, AI Written, or AI Written (Humanized).")

    # Paths to the model and tokenizer
    model_path = 'rnn.h5'
    tokenizer_path = 'tokenizer.pkl'

    # Load the model and tokenizer
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    if tokenizer is None:
        st.error("Failed to load tokenizer. Please check the tokenizer file.")
        return

    # Input text box
    text = st.text_area("Enter text here:", height=150)

    # Evaluate button
    if st.button("Classify"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Evaluate the text
            predicted_class, confidence_scores = evaluate_text(text, model, tokenizer)
            
            # Display results
            st.success(f"**Predicted Class:** {predicted_class}")
            st.write("**Confidence Scores:**")
            for label, score in confidence_scores.items():
                st.write(f"- {label}: {score:.4f}")

# Run the Streamlit app
if __name__ == '__main__':
    main()