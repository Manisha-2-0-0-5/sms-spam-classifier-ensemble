import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging
import warnings
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Download NLTK data
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        logging.error(f"NLTK setup failed: {str(e)}")
        return False

setup_nltk()

# Set up Streamlit page
st.set_page_config(
    page_title="SMS Spam Classifier with Ensemble", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with fallback for gradient compatibility
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        /* Fallback color if gradient fails */
        color: #4B0082;
        /* Gradient with fallback */
        background: -webkit-linear-gradient(45deg, #4B0082, #9370DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4B0082;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4B0082;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: #f5f7fa; /* Fallback solid color */
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .prediction-spam {
        background: #ff4d4d; /* Fallback solid color */
        background: linear-gradient(135deg, #ff4d4d 0%, #cc0000 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
    }
    .prediction-ham {
        background: #4CAF50; /* Fallback solid color */
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
    }
    .stProgress > div > div > div > div {
        background: #4B0082; /* Fallback solid color */
        background: linear-gradient(90deg, #4B0082 0%, #9370DB 100%);
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4B0082;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 20px;
        margin-top: 30px;
        border-top: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Text Preprocessor Class with enhanced features
class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'])
            self.stemmer = None
    
    def preprocess(self, text):
        if not text or not isinstance(text, str):
            logging.warning("Invalid input text")
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', 'url', text)
        text = re.sub(r'\d+', 'number', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        if self.stemmer:
            words = [self.stemmer.stem(word) for word in words]
        return ' '.join(words)
    
    def extract_features(self, text):
        if not text or not isinstance(text, str):
            return {'length': 0, 'has_url': 0, 'num_special_chars': 0, 'num_digits': 0}
        
        # Count special characters
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        
        # Count digits
        digits = len(re.findall(r'\d', text))
        
        # Count uppercase letters (if any remain after lowercasing)
        uppercase = len(re.findall(r'[A-Z]', text))
        
        return {
            'length': len(text),
            'has_url': int(bool(re.search(r'http\S+|www\.\S+', text))),
            'num_special_chars': special_chars,
            'num_digits': digits,
            'uppercase_ratio': uppercase / max(1, len(text))
        }

# Load dataset (automatically download UCI if not present)
@st.cache_data
def load_dataset():
    dataset_path = 'SMSSpamCollection'
    if not os.path.exists(dataset_path):
        st.info("Downloading UCI SMS Spam Collection dataset...")
        try:
            import urllib.request
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip", "smsspamcollection.zip")
            import zipfile
            with zipfile.ZipFile("smsspamcollection.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("smsspamcollection.zip")
            st.success("Dataset downloaded successfully!")
        except Exception as e:
            logging.error(f"Download failed: {str(e)}")
            st.error("Download failed. Using synthetic data instead.")
            texts = [
                "Win $1000 now! Click www.example.com!",
                "Urgent: Verify account at bank.com!",
                "Free offer! Text YES to 12345!",
                "Claim your prize at win.fake.com!",
                "90% off sale! Visit shop.now.com!",
                "Hey, lunch at 1 PM?",
                "Can you grab milk?",
                "Meeting at 2 PM.",
                "Thanks for the help!",
                "Report sent.",
                "Happy birthday!",
                "Running late 10 mins.",
                "See you tomorrow!",
                "Weather looks good.",
                "Congrats on promotion!",
                "Great presentation!",
                "Mom's birthday Sunday.",
                "Reschedule dinner?",
                "Flight delayed to 7 PM.",
                "Loved the book!"
            ]
            labels = [1] * 5 + [0] * 15
            return pd.DataFrame({'text': texts, 'label': labels})
    
    try:
        df = pd.read_csv(dataset_path, sep='\t', names=['label', 'text'])
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        st.error("Could not load dataset. Using synthetic data instead.")
        texts = [
            "Win $1000 now! Click www.example.com!",
            "Urgent: Verify account at bank.com!",
            "Free offer! Text YES to 12345!",
            "Claim your prize at win.fake.com!",
            "90% off sale! Visit shop.now.com!",
            "Hey, lunch at 1 PM?",
            "Can you grab milk?",
            "Meeting at 2 PM.",
            "Thanks for the help!",
            "Report sent.",
            "Happy birthday!",
            "Running late 10 mins.",
            "See you tomorrow!",
            "Weather looks good.",
            "Congrats on promotion!",
            "Great presentation!",
            "Mom's birthday Sunday.",
            "Reschedule dinner?",
            "Flight delayed to 7 PM.",
            "Loved the book!"
        ]
        labels = [1] * 5 + [0] * 15
        return pd.DataFrame({'text': texts, 'label': labels})

# Train ensemble model
@st.cache_resource
def train_model():
    try:
        df = load_dataset()
        preprocessor = TextPreprocessor()
        
        # Add progress bar for preprocessing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Preprocessing text...")
        df['processed_text'] = df['text'].apply(preprocessor.preprocess)
        progress_bar.progress(30)
        
        status_text.text("Extracting features...")
        feature_data = [preprocessor.extract_features(text) for text in df['text']]
        feature_df = pd.DataFrame(feature_data)
        progress_bar.progress(50)
        
        status_text.text("Vectorizing text...")
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        X_tfidf = vectorizer.fit_transform(df['processed_text'])
        progress_bar.progress(70)
        
        X = np.hstack([X_tfidf.toarray(), feature_df.values])
        y = df['label'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        status_text.text("Training models...")
        xgb_model = XGBClassifier(
            n_estimators=50, max_depth=3, 
            scale_pos_weight=sum(y==0)/sum(y==1), 
            random_state=42, eval_metric='logloss'
        )
        lr_model = LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
        nb_model = MultinomialNB(alpha=0.5)
        
        ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lr', lr_model),
                ('nb', nb_model)
            ],
            voting='soft', weights=[2, 1, 1]
        )
        
        ensemble_model.fit(X_train, y_train)
        progress_bar.progress(90)
        
        status_text.text("Evaluating model...")
        y_pred = ensemble_model.predict(X_test)
        y_pred_proba = ensemble_model.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'predictions': y_pred,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return ensemble_model, vectorizer, metrics, df, preprocessor
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

# Function to create word clouds
def create_wordclouds(df):
    spam_text = " ".join(df[df['label'] == 1]['processed_text'].tolist())
    ham_text = " ".join(df[df['label'] == 0]['processed_text'].tolist())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Spam word cloud
    spam_wc = WordCloud(width=500, height=300, background_color='white', colormap='Reds').generate(spam_text)
    ax1.imshow(spam_wc, interpolation='bilinear')
    ax1.set_title('Spam Messages Word Cloud', color='red')
    ax1.axis('off')
    
    # Ham word cloud
    ham_wc = WordCloud(width=500, height=300, background_color='white', colormap='Greens').generate(ham_text)
    ax2.imshow(ham_wc, interpolation='bilinear')
    ax2.set_title('Ham Messages Word Cloud', color='green')
    ax2.axis('off')
    
    return fig

# App title
st.markdown('<h1 class="main-header">SMS Spam Classifier with Ensemble</h1>', unsafe_allow_html=True)
st.write("A powerful tool to detect spam SMS messages using an ensemble of XGBoost, Logistic Regression, and Naive Bayes.")

# Load model
with st.spinner("Training model... This may take a moment."):
    model, vectorizer, metrics, df, preprocessor = train_model()
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        model_loaded = False
    else:
        model_loaded = True

if model_loaded:
    # Sidebar with enhanced options
    st.sidebar.markdown("## Performance Metrics")
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>F1-Score:</strong> {metrics['f1_score']*100:.2f}%<br>
        <strong>Accuracy:</strong> {metrics['accuracy']*100:.2f}%
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("## Dataset Info")
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Total Messages:</strong> {len(df)}<br>
        <strong>Spam Messages:</strong> {sum(df['label'])} ({sum(df['label'])/len(df)*100:.1f}%)<br>
        <strong>Ham Messages:</strong> {len(df) - sum(df['label'])}
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    options = st.sidebar.radio("Select Section:", ["Home", "Classify", "Results", "Dataset Analysis"])
    
    if options == "Home":
        st.markdown("## Welcome to the SMS Spam Classifier")
        st.write("""
        This tool uses an ensemble of three machine learning models (XGBoost, Logistic Regression, and Naive Bayes) 
        to detect spam SMS messages. Enter a message in the 'Classify' section to check if it's spam or ham.
        The 'Results' section shows the model's performance, and 'Dataset Analysis' provides insights into the data.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### How It Works")
            st.markdown("""
            <div class="feature-box">
            1. <strong>Text Preprocessing:</strong> Messages are cleaned, tokenized, and stemmed
            </div>
            <div class="feature-box">
            2. <strong>Feature Extraction:</strong> TF-IDF vectors and custom features are extracted
            </div>
            <div class="feature-box">
            3. <strong>Ensemble Classification:</strong> Multiple models vote on the final prediction
            </div>
            <div class="feature-box">
            4. <strong>Results:</strong> Confidence scores and explanations are provided
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Model Ensemble Details")
            st.markdown("""
            <div class="feature-box">
            <strong>XGBoost:</strong> Gradient boosting algorithm with 2x weight in voting
            </div>
            <div class="feature-box">
            <strong>Logistic Regression:</strong> Linear model with balanced class weights
            </div>
            <div class="feature-box">
            <strong>Naive Bayes:</strong> Probabilistic classifier with smoothing
            </div>
            <div class="feature-box">
            <strong>Voting:</strong> Soft voting with weighted probabilities
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Try It Out")
        st.write("Navigate to the 'Classify' section to test messages, or explore the 'Results' and 'Dataset Analysis' sections for more insights.")
        
    elif options == "Classify":
        st.markdown("## Classify an SMS Message")
        
        # Example messages
        example_messages = [
            "Win $1000 now! Click www.example.com!",
            "Hey, are we still meeting for lunch tomorrow?",
            "URGENT: Your bank account needs verification. Click here: bank-verify.com",
            "Your package has been delivered. Track at: delivery-track.com/12345"
        ]
        
        selected_example = st.selectbox("Or choose an example message:", 
                                       ["Select an example..."] + example_messages)
        
        if selected_example != "Select an example...":
            user_input = st.text_area("Enter SMS message:", selected_example, height=100)
        else:
            user_input = st.text_area("Enter SMS message:", "Win $1000 now! Click www.example.com!", height=100)
        
        if st.button("Check Message", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing message..."):
                    processed_text = preprocessor.preprocess(user_input)
                    features = preprocessor.extract_features(user_input)
                    tfidf_features = vectorizer.transform([processed_text])
                    combined_features = np.hstack([tfidf_features.toarray(), np.array([list(features.values())])])
                    
                    prediction = model.predict(combined_features)[0]
                    confidence = model.predict_proba(combined_features)[0][prediction] * 100
                    probabilities = model.predict_proba(combined_features)[0]
                    
                    if prediction == 1:
                        st.markdown(f'<div class="prediction-spam">🚨 SPAM DETECTED! (Confidence: {confidence:.1f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-ham">✅ LEGITIMATE MESSAGE (Confidence: {confidence:.1f}%)</div>', unsafe_allow_html=True)
                    
                    # Show probability distribution
                    fig = go.Figure(data=[
                        go.Bar(x=['Ham', 'Spam'], y=probabilities, 
                               marker_color=['#4CAF50', '#FF4D4D'])
                    ])
                    fig.update_layout(
                        title='Prediction Probabilities',
                        yaxis=dict(range=[0, 1], title='Probability'),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show extracted features
                    st.markdown("### Message Features")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="feature-box">
                            <strong>Length:</strong> {features['length']} characters<br>
                            <strong>Contains URL:</strong> {'Yes' if features['has_url'] else 'No'}<br>
                            <strong>Special Characters:</strong> {features['num_special_chars']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="feature-box">
                            <strong>Digits:</strong> {features['num_digits']}<br>
                            <strong>Uppercase Ratio:</strong> {features['uppercase_ratio']:.3f}<br>
                            <strong>Processed Text:</strong> {processed_text[:50]}{'...' if len(processed_text) > 50 else ''}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a valid message.")
                
    elif options == "Results":
        st.markdown("## Model Performance")
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(metrics['y_test'], metrics['predictions'])
        fig = px.imshow(
            cm, text_auto=True, aspect="auto", color_continuous_scale='Blues',
            title='Confusion Matrix', labels=dict(x="Predicted", y="Actual"), 
            x=['Ham', 'Spam'], y=['Ham', 'Spam']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='#d4f1d4'))
        
        # ROC Curve (simulated)
        st.markdown("### Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <strong>Accuracy</strong><br>
                {metrics['accuracy']*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <strong>F1 Score</strong><br>
                {metrics['f1_score']*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate precision and recall from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>Precision</strong><br>
                {precision*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
        
    elif options == "Dataset Analysis":
        st.markdown("## Dataset Analysis")
        
        # Distribution of labels
        st.markdown("### Message Distribution")
        label_counts = df['label'].value_counts()
        fig = px.pie(values=label_counts.values, names=['Ham', 'Spam'], 
                     title='Proportion of Spam vs Ham Messages',
                     color=['Ham', 'Spam'], color_discrete_map={'Ham':'#4CAF50', 'Spam':'#FF4D4D'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Message length analysis
        st.markdown("### Message Length Analysis")
        df['message_length'] = df['text'].apply(len)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Ham Messages', 'Spam Messages'))
        
        ham_lengths = df[df['label'] == 0]['message_length']
        spam_lengths = df[df['label'] == 1]['message_length']
        
        fig.add_trace(go.Histogram(x=ham_lengths, name='Ham', marker_color='#4CAF50'), row=1, col=1)
        fig.add_trace(go.Histogram(x=spam_lengths, name='Spam', marker_color='#FF4D4D'), row=1, col=2)
        
        fig.update_xaxes(title_text='Message Length', row=1, col=1)
        fig.update_xaxes(title_text='Message Length', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Word clouds
        st.markdown("### Word Frequency Analysis")
        with st.spinner("Generating word clouds..."):
            wordcloud_fig = create_wordclouds(df)
            st.pyplot(wordcloud_fig)

# Footer
st.markdown("""
<div class="footer">
    <h4>SMS Spam Classifier with Ensemble</h4>
    <p>Built for college project using Python, Scikit-learn, XGBoost, NLTK, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
