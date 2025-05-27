import sys

import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

# Rest of imports
import os
import time
import gdown
import torch
import streamlit as st
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, pipeline

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["PYTORCH_JIT"] = "0"


# --------------------------
# STREAMLIT CONFIG
# --------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# --------------------------
# MODEL LOADING (Cached)
# --------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Download custom model weights from Google Drive
        url = f"https://drive.google.com/file/d/1u-GQq8Ei4_-ll5WOfb1XaQ3QRPsXAjR7/view?usp=sharing"
        output_path = "c1_fakenews_weights.pt"
        
        if not os.path.exists(output_path):
            with st.spinner("📥 Downloading model weights from Google Drive..."):
                gdown.download(url, output_path, quiet=True)

        # Load BERT base model
        bert = AutoModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        # Define model architecture
        class BERT_Arch(nn.Module):
            def __init__(self, bert):
                super().__init__()
                self.bert = bert
                self.dropout = nn.Dropout(0.1)
                self.fc1 = nn.Linear(768, 512)
                self.fc2 = nn.Linear(512, 2)
                self.softmax = nn.LogSoftmax(dim=1)
            
            def forward(self, sent_id, mask):
                outputs = self.bert(sent_id, attention_mask=mask)
                cls_hs = outputs.last_hidden_state[:, 0, :]
                x = self.fc1(cls_hs)
                x = torch.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return self.softmax(x)
        
        # Initialize and load weights
        model = BERT_Arch(bert)
        model.load_state_dict(torch.load(output_path, map_location='cpu'), strict=False)
        model.eval()

        # Load pretrained pipeline model
        pipe_model = pipeline("text-classification", model="tvocoder/bert_fake_news_ft")
        
        return model, tokenizer, pipe_model

    except Exception as e:
        st.error(f"🚨 Model loading failed: {str(e)}")
        st.stop()

# Initialize models
model, tokenizer, pipe_model = load_models()

# --------------------------
# STREAMLIT UI COMPONENTS
# --------------------------
# Custom styling
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .stTextArea textarea { 
        font-size: 16px !important; 
        padding: 10px !important; 
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
    }
    .stButton>button { 
        background: #4CAF50 !important; 
        color: white !important; 
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: scale(1.05);
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .model-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .comparison-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 Fake News Detection System")
st.markdown("---")

# Example selector
examples = [
    "Select an example...",
    "Atif Aslam is a world famous Pakistani singer",  # Fake
    "America got its independence from Britain on July 4th",  # Real
    "5G networks spread coronavirus",  # Fake
    "NASA confirms water on Mars surface"  # Real
]

selected_example = st.selectbox("Try a sample text:", examples)

# Text input
input_text = st.text_area(
    "**Enter news text to analyze:**",
    height=150,
    value=selected_example if selected_example != examples[0] else ""
)

# Prediction logic
if st.button("🔎 Analyze Text", use_container_width=True):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        with st.spinner("🔍 Analyzing with both models..."):
            # Original model processing
            start_time1 = time.time()
            tokens = tokenizer(
                input_text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                logits = model(tokens['input_ids'], tokens['attention_mask'])
                probs = torch.exp(logits).numpy()[0]
            
            original_time = time.time() - start_time1
            fake_prob1 = probs[0] * 100
            real_prob1 = probs[1] * 100

            # Pipeline model processing
            start_time2 = time.time()
            result = pipe_model(input_text)[0]
            pipe_time = time.time() - start_time2
            label = "FAKE" if result['label'] == "FAKE" else "REAL"
            confidence = result['score'] * 100
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.subheader("🧠 Fine-tuned BERT")
                st.markdown(f'<div class="metric-box">⏱️ Time: {original_time:.2f}s</div>', unsafe_allow_html=True)
                st.success(f"✅ Real: {real_prob1:.1f}%") if real_prob1 > 50 else st.error(f"❌ Fake: {fake_prob1:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.subheader("🚀 Pretrained BERT")
                st.markdown(f'<div class="metric-box">⏱️ Time: {pipe_time:.2f}s</div>', unsafe_allow_html=True)
                st.success(f"✅ {label}: {confidence:.1f}%") if label == "REAL" else st.error(f"❌ {label}: {confidence:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            # Comparison section
            st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
            st.subheader("📊 Performance Comparison")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.metric("Confidence Difference", f"{abs(real_prob1 - confidence):.1f}%")
            
            with comp_col2:
                st.metric("Speed Difference", f"{abs(original_time - pipe_time):.2f}s")
            
            with comp_col3:
                fastest = "Pretrained" if pipe_time < original_time else "Fine-tuned"
                st.metric("Fastest Model", fastest)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Built with 🤗 Transformers and Streamlit | Fake News Detection System")
