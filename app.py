import gradio as gr
import numpy as np
import joblib
from urllib.parse import urlparse
from url_features import extract_features

# load model and preprocessing
model = joblib.load("model/phishing_model.pkl")
scaler = joblib.load("model/scaler.pkl")
pca = joblib.load("model/pca.pkl")


def predict_url(url):

    url_lower = url.lower()

    # ---------- RULE BASED DETECTION ----------
    phishing_keywords = [
        "login","verify","secure","update","account",
        "bank","paypal","signin","confirm","security"
    ]

    suspicious_tlds = [
        ".xyz",".ru",".tk",".ml",".ga",".cf"
    ]

    # keyword check
    if any(word in url_lower for word in phishing_keywords):
        result = "⚠️ Phishing Website"
        confidence = "High (Rule Based)"
    elif any(tld in url_lower for tld in suspicious_tlds):
        result = "⚠️ Phishing Website"
        confidence = "High (Suspicious Domain)"
    else:

        # ---------- MACHINE LEARNING ----------
        features = extract_features(url)
        features = np.array(features).reshape(1, -1)

        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        prediction = model.predict(features_pca)
        probability = model.predict_proba(features_pca)

        if prediction[0] == 0:
            result = "✅ Legitimate Website"
            confidence = f"{probability[0][0]*100:.2f}%"
        else:
            result = "⚠️ Phishing Website"
            confidence = f"{probability[0][1]*100:.2f}%"

    # ---------- FEATURE ANALYSIS ----------
    parsed = urlparse(url)
    domain = parsed.netloc

    dots = url.count(".")
    dash = url.count("-")
    length = len(url)
    https = "Yes" if "https" in url else "No"

    analysis = f"""
Domain: {domain}

URL Length: {length}
Number of Dots: {dots}
Number of Dash: {dash}
HTTPS Used: {https}
"""

    # ---------- RISK SCORE ----------
    risk = 0

    if length > 75:
        risk += 20

    if dash > 2:
        risk += 20

    if "login" in url_lower:
        risk += 30

    if "secure" in url_lower:
        risk += 20

    risk_score = f"{risk}/100"

    return result, confidence, risk_score, analysis


interface = gr.Interface(
    fn=predict_url,
    inputs=gr.Textbox(label="Enter Website URL"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Text(label="Confidence Score"),
        gr.Text(label="Risk Score"),
        gr.Textbox(label="Feature Analysis")
    ],
    title="🛡 Phishing Website Detection System",
    description="Enter a URL to check if the website is phishing or legitimate.",
    theme="soft"
)

interface.launch()