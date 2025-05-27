# 1. Environment setup MUST COME FIRST
import os
import time
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# 2. PyTorch imports and workaround
import torch
torch.__streamlit__ = False  # Block Streamlit's class inspection

# 3. Other imports
import gdown
import streamlit as st
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, pipeline
# app.py
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

@st.cache_resource
def load_model():
    # Download from Google Drive
    url = "https://drive.google.com/file/d/1u-GQq8Ei4_-ll5WOfb1XaQ3QRPsXAjR7/view?usp=drive_link"
    gdown.download(url, "c1_fakenews_weights.pt", quiet=False)
    
    # Load model
    model.load_state_dict(torch.load("c1_fakenews_weights.pt"))
    return model

# -------------------------


def load_models():
    # Load original custom model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
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
            x = self.softmax(x)
            return x
    
    model = BERT_Arch(bert)
    
    # Load state_dict with strict=False to ignore unexpected keys
    state_dict = torch.load(
        r'c1_fakenews_weights.pt',
        map_location=torch.device('cpu')
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load pretrained pipeline model
    pipe_model = pipeline("text-classification", model="tvocoder/bert_fake_news_ft")
    
    return model, tokenizer, pipe_model

model, tokenizer, pipe_model = load_models()

# Rest of your UI code remains the same...
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .stTextArea textarea { 
        font-size: 16px !important; 
        padding: 10px !important; 
        border-radius: 8px !important;
    }
    .stButton>button { 
        background: #4CAF50 !important; 
        color: white !important; 
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
    }
    .model-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
st.title("🔍 Fake News Detector")
st.markdown("---")

# Example selector
examples = [
    "Select an example...",
    "Atif Aslam is a world famous Pakistani singer",  # Fake
    "America got it's freedom from British on 4th of July",              # Real
    "World leaders sign global climate agreement",        # Real
    "5G networks spread coronavirus"                      # Fake
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
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing with both models..."):
            # Original model processing
            start_time1 = time.time()
            tokens = tokenizer.batch_encode_plus(
                [input_text],
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
                st.subheader("🧠 Finetuned BERT Model")
                st.markdown(f'<div class="metric-box">⏱️ Inference Time: {original_time:.2f}s</div>', unsafe_allow_html=True)
                if real_prob1 > fake_prob1:
                    st.success(f"✅ Real News ({real_prob1:.1f}% confidence)")
                else:
                    st.error(f"❌ Fake News ({fake_prob1:.1f}% confidence)")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.subheader("🚀 Pretrained Bert Model")
                st.markdown(f'<div class="metric-box">⏱️ Inference Time: {pipe_time:.2f}s</div>', unsafe_allow_html=True)
                if label == "REAL":
                    st.success(f"✅ {label} News ({confidence:.1f}% confidence)")
                else:
                    st.error(f"❌ {label} News ({confidence:.1f}% confidence)")
                st.markdown('</div>', unsafe_allow_html=True)

            # Comparison section
            st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
            st.subheader("📊 Model Comparison")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown(f'<div class="metric-box">\
                    🏆 Confidence Difference<br>\
                    <h3>{abs(real_prob1 - confidence):.1f}%</h3>\
                    <small>Custom vs Pipeline</small></div>', 
                    unsafe_allow_html=True)
            
            with comp_col2:
                st.markdown(f'<div class="metric-box">\
                    ⚡ Speed Difference<br>\
                    <h3>{abs(original_time - pipe_time):.2f}s</h3>\
                    <small>Custom vs Pipeline</small></div>', 
                    unsafe_allow_html=True)
            
            with comp_col3:
                winner = "Pipeline" if pipe_time < original_time else "Custom"
                st.markdown(f'<div class="metric-box">\
                    🏅 Faster Model<br>\
                    <h3>{winner}</h3>\
                    <small>Based on inference time</small></div>', 
                    unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Built with 🤗 Transformers and Streamlit | Fake News Detection System")
