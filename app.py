import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer — Twitter",
    page_icon="🐦",
    layout="centered",
)

# ── Cargar modelo (se cachea para no recargar en cada interacción) ────────────
@st.cache_resource
def load_artifact():
    return joblib.load("modelo_final.joblib")

artifact   = load_artifact()
model      = artifact["model"]
model_name = artifact["model_name"]
tfidf      = artifact["tfidf"]
scaler     = artifact["scaler"]
svd        = artifact["svd"]
classes    = artifact["classes"]
tab_cols   = artifact["tab_columns"]

NEGATION_WORDS = {
    "not", "never", "no", "neither",
    "don't", "didn't", "won't", "can't", "isn't", "wasn't",
}

CLASS_COLORS = {
    "Positive":   "#2ecc71",
    "Negative":   "#e74c3c",
    "Neutral":    "#3498db",
    "Irrelevant": "#95a5a6",
}

CLASS_EMOJI = {
    "Positive":   "😊",
    "Negative":   "😠",
    "Neutral":    "😐",
    "Irrelevant": "🤷",
}

def extract_features(text: str) -> dict:
    words      = text.split()
    lengths    = [len(w) for w in words] if words else [0]
    n_upper    = sum(1 for c in text if c.isupper())
    n_alpha    = sum(1 for c in text if c.isalpha())
    return {
        "tweet_length":    len(text),
        "word_count":      len(words),
        "avg_word_length": float(np.mean(lengths)),
        "n_exclamations":  text.count("!"),
        "n_questions":     text.count("?"),
        "n_uppercase":     n_upper,
        "uppercase_ratio": n_upper / max(n_alpha, 1),
        "n_hashtags":      text.count("#"),
        "n_mentions":      text.count("@"),
        "n_urls":          len(re.findall(r"http\S+", text)),
        "has_negation":    int(any(w.lower() in NEGATION_WORDS for w in words)),
    }

def predict(tweet_text: str):
    X_tfidf = tfidf.transform([tweet_text])

    tab = pd.DataFrame([extract_features(tweet_text)])
    for col in tab_cols:
        if col not in tab.columns:
            tab[col] = 0.0
    tab = tab[tab_cols]
    tab_sc = scaler.transform(tab)

    X_svd  = svd.transform(X_tfidf)
    svd_cols = [f"svd_{i}" for i in range(X_svd.shape[1])]
    X = pd.DataFrame(
        np.hstack([X_svd, tab_sc]),
        columns=svd_cols + tab_cols,
    )

    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    return pred, dict(zip(model.classes_, proba))

# ── Interfaz ──────────────────────────────────────────────────────────────────
st.title("🐦 Twitter Sentiment Analyzer")
st.caption(
    f"Modelo: **{model_name}** · "
    "Clasificación multiclase: Positive / Negative / Neutral / Irrelevant"
)
st.divider()

tweet_input = st.text_area(
    "Ingresá un tweet en inglés:",
    placeholder="Ej: I absolutely love this new update! Best thing ever!",
    height=100,
)

predict_btn = st.button("Predecir sentimiento", type="primary", use_container_width=True)

if predict_btn:
    text = tweet_input.strip()
    if not text:
        st.warning("Escribí un tweet antes de predecir.")
    else:
        with st.spinner("Analizando..."):
            pred, proba = predict(text)

        color = CLASS_COLORS[pred]
        emoji = CLASS_EMOJI[pred]

        st.divider()

        # Resultado principal
        st.markdown(
            f"<div style='background:{color}22; border-left:5px solid {color};"
            f"padding:16px 20px; border-radius:8px; margin-bottom:16px'>"
            f"<span style='font-size:2rem'>{emoji}</span>&nbsp;&nbsp;"
            f"<span style='font-size:1.4rem; font-weight:700; color:{color}'>{pred}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Gráfico de probabilidades
        st.subheader("Probabilidades por clase")

        ordered = sorted(proba.items(), key=lambda x: x[1], reverse=True)
        labels  = [c for c, _ in ordered]
        values  = [v for _, v in ordered]
        colors  = [CLASS_COLORS[c] for c in labels]

        fig, ax = plt.subplots(figsize=(6, 2.8))
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilidad")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        for bar, val in zip(bars, values[::-1]):
            ax.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10,
            )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "⚠️ La entidad (marca) no se ingresa en esta demo — "
            "las features Entity se imputan como neutras. "
            "La predicción se basa principalmente en el texto del tweet."
        )

st.divider()
st.caption(
    "Proyecto: Twitter Entity Sentiment Analysis · "
    "Estadística 98 para Ingeniería Informática · "
    "Entrenado con LightGBM + TF-IDF + features tabulares · F1-macro test: 0.7515"
)
