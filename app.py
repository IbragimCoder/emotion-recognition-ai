import streamlit as st
import joblib
import pandas as pd
import re
import string

@st.cache_resource
def load_model_pipeline():
    return joblib.load('emotion_model.joblib')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def is_valid_input(text):
    if len(text.strip()) < 2:
        return False, "The text is too short."
        
    if not re.search(r'[a-zA-Z]', text):
        return False, "Use English."
        
    if not re.search(r'[aeiouyAEIOUY]', text):
        return False, "The word does not look like English (there are no vowels)."
        
    return True, ""

pipeline = load_model_pipeline()

emotion_map = {
    0: 'sadness 😢',
    1: 'joy 😃',
    2: 'love 🥰',
    3: 'anger 😡',
    4: 'fear 😱',
    5: 'surprise 😲'
}

st.title("🔍 AI Emotion Analyzer")
st.markdown("""
Enter the text in English, and the model will identify the emotion.
*Examples: "I am happy today", "I feel nervous about the interview"*
""")

user_input = st.text_area("Your text:", height=100)

if st.button("Identify an emotion"):
    is_valid, error_message = is_valid_input(user_input)
    
    if not is_valid:
        st.warning(f"⚠️ {error_message}")
    else:
        cleaned_input = clean_text(user_input)
        
        if not cleaned_input:
             st.warning("⚠️ The text does not contain significant words.")
        else:
            prediction_idx = pipeline.predict([cleaned_input])[0]
            prediction_prob = pipeline.predict_proba([cleaned_input])[0]
            
            confidence = prediction_prob[prediction_idx]
            
            THRESHOLD = 0.40
            
            if confidence < THRESHOLD:
                st.warning(f"🤔 Model is not sure (Confidence: {confidence:.2%}). Perhaps the text is too complicated or meaningless.")
            else:
                predicted_label = emotion_map[prediction_idx]
                
                if prediction_idx in [1, 2, 5]:
                    st.success(f"Result: **{predicted_label}**")
                else:
                    st.error(f"Result: **{predicted_label}**")
                
                st.info(f"Confidence level: {confidence:.2%}")

            with st.expander("View details (probabilities)"):
                prob_df = pd.DataFrame(
                    prediction_prob, 
                    index=emotion_map.values(), 
                    columns=["Probability"]
                )
                st.bar_chart(prob_df)