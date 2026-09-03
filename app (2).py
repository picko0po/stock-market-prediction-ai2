
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Market Prediction")
st.write(
    "Comparison of LSTM and Random Forest models "
    "for stock-return prediction."
)

# -----------------------------
# Model results
# -----------------------------

results = pd.DataFrame({
    "Model": [
        "Jin Choi LSTM",
        "Random Forest",
        "Improved LSTM"
    ],
    "R²": [
        -0.0015,
        0.0082,
        0.0004
    ],
    "MAE": [
        0.0083,
        0.014117,
        0.013448
    ],
    "RMSE": [
        None,
        0.019845,
        0.019533
    ],
    "Direction Accuracy": [
        None,
        0.5447,
        0.4770
    ]
})

# -----------------------------
# Results table
# -----------------------------

st.header("Model Performance")

st.dataframe(
    results,
    use_container_width=True
)

# -----------------------------
# R² chart
# -----------------------------

st.header("R² Comparison")

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    results["Model"],
    results["R²"]
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_ylabel("R²")
ax.set_title("Stock Prediction Model Comparison")

plt.xticks(rotation=15)
plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Key finding
# -----------------------------

st.header("Current Finding")

st.write(
    "Random Forest currently has the highest R² "
    "among the three recorded models."
)

st.info(
    "These results are experimental. "
    "A near-zero R² means the models have limited "
    "ability to explain next-day stock-return variation."
)

st.caption(
    "Baseline approach inspired by Jin Choi's publicly available work. "
    "This project extends the baseline with multiple stocks, "
    "additional features, Random Forest, and an improved LSTM."
)
