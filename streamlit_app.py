import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Cloud IDS Dashboard",
    layout="wide"
)

# ==================================================
# CUSTOM CSS (SIDEBAR + KPI CARDS)
# ==================================================
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #E6EDF3;
}

section[data-testid="stSidebar"] {
    background-color: #0B0F14;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #E6EDF3 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    font-size: 16px !important;
    color: #C9D1D9 !important;
}

/* KPI CARDS */
.kpi-row {
    display: flex;
    gap: 20px;
    margin-top: 20px;
}

.kpi-card {
    background-color: #161B22;
    border-radius: 10px;
    width: 100%;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.kpi-header {
    background-color: #1F2937;
    color: #E5E7EB;
    font-size: 13px;
    font-weight: 700;
    padding: 8px 14px;
    text-transform: uppercase;
}

.kpi-body {
    padding: 18px 14px;
    font-size: 28px;
    font-weight: 700;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL ARTIFACTS
# ==================================================
MODEL = tf.keras.models.load_model("ids_model.keras")
SCALER = joblib.load("scaler.pkl")
THRESHOLD = float(np.load("best_threshold.npy"))

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 🔐 Cloud IDS")
    st.markdown("**Post-Quantum Encrypted Logs**")
    st.markdown("---")

    file = st.file_uploader(
        "Upload CICIDS Dataset (CSV)",
        type=["csv"]
    )

    st.markdown("---")
    menu = st.radio(
        "Navigation",
        ["Dashboard", "Alerts", "Metrics"]
    )

# ==================================================
# STOP IF NO FILE
# ==================================================
if not file:
    st.info("Please upload a CICIDS CSV file from the left panel.")
    st.stop()

# ==================================================
# DATA LOADING
# ==================================================
df = pd.read_csv(file)

# ==================================================
# PREPROCESSING
# ==================================================
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Protocol"]
df.drop(columns=drop_cols, errors="ignore", inplace=True)

original_ports = df["Destination Port"].values
X = df.drop("Label", axis=1)
y = LabelEncoder().fit_transform(df["Label"])

X_scaled = SCALER.transform(X)

# ==================================================
# POST-QUANTUM ENCRYPTION (CLIENT)
# ==================================================
key = np.random.bytes(32)   # 256-bit PQ-safe key
iv = np.random.bytes(16)

cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
encryptor = cipher.encryptor()
encrypted_logs = encryptor.update(X_scaled.tobytes()) + encryptor.finalize()

# ==================================================
# DECRYPTION (CLOUD)
# ==================================================
decryptor = cipher.decryptor()
decrypted = decryptor.update(encrypted_logs) + decryptor.finalize()
X_cloud = np.frombuffer(decrypted, dtype=X_scaled.dtype).reshape(X_scaled.shape)

# ==================================================
# IDS INFERENCE
# ==================================================
probs = MODEL.predict(X_cloud, verbose=0).ravel()
preds = (probs >= THRESHOLD).astype(int)

# ==================================================
# ALERT GENERATION
# ==================================================
alerts = []
for i, p in enumerate(probs):
    if p >= THRESHOLD:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Packet ID": i,
            "Destination Port": int(original_ports[i]),
            "Confidence": round(float(p), 4),
            "Severity": "HIGH" if p >= 0.95 else "MEDIUM"
        })

alerts_df = pd.DataFrame(alerts)

# ==================================================
# DASHBOARD
# ==================================================
if menu == "Dashboard":
    st.markdown("## ☁️ Cloud Intrusion Detection Dashboard")
    st.caption("Encrypted Traffic • Post-Quantum Security • Deep Learning IDS")

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    accuracy = accuracy_score(y, preds) * 100
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-header">Total Logs</div>
            <div class="kpi-body">{len(df):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">Detected Attacks</div>
            <div class="kpi-body">{len(alerts_df):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">Accuracy</div>
            <div class="kpi-body">{accuracy:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">False Positive Rate</div>
            <div class="kpi-body">{fpr:.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([3, 1.5])

    with left:
        st.subheader("🎯 Top Targeted Destination Ports")
        st.bar_chart(alerts_df["Destination Port"].value_counts().head(10))

    with right:
        st.subheader("🔄 Security Pipeline")
        st.markdown("""
        **Client Side**
        - Log collection  
        - Encrypted using **Post-Quantum 256-bit key**

        **Cloud Side**
        - Secure reception  
        - AES-CFB decryption  

        **Detection**
        - Deep Learning IDS  
        - Alert generation
        """)

# ==================================================
# ALERTS PAGE
# ==================================================
if menu == "Alerts":
    st.subheader("🚨 Intrusion Alerts")
    st.dataframe(alerts_df, use_container_width=True)
    st.download_button(
        "⬇️ Download Alert Log",
        alerts_df.to_csv(index=False),
        "alert_log.csv",
        "text/csv"
    )

# ==================================================
# METRICS PAGE
# ==================================================
if menu == "Metrics":
    st.subheader("📊 Evaluation Metrics")

    metrics = {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1 Score": f1_score(y, preds),
        "False Positives": int(fp),
        "False Negatives": int(fn)
    }

    st.json(metrics)

    st.subheader("📉 Confusion Matrix")
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        columns=["Predicted Normal", "Predicted Attack"],
        index=["Actual Normal", "Actual Attack"]
    )
    st.dataframe(cm_df, use_container_width=True)
