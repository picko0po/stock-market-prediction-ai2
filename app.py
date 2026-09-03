import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #07111f;
    }

    section[data-testid="stSidebar"] {
        background: #0b1726;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #aebbd0;
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(145deg, #172333, #101b29);
        border: 1px solid #26374b;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        min-height: 125px;
    }

    .metric-title {
        color: #c8d2df;
        font-size: 14px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }

    .metric-model {
        color: #aebbd0;
        font-size: 13px;
        margin-top: 4px;
    }

    .prediction-box {
        background: linear-gradient(145deg, #152638, #0e1a29);
        border: 1px solid #2d435c;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
    }

    .prediction-up {
        color: #45d483;
        font-size: 38px;
        font-weight: 800;
    }

    .prediction-down {
        color: #ff5b62;
        font-size: 38px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

SEED = 42
WINDOW = 20
TEST_RATIO = 0.20

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# ============================================================
# FIND CSV FILE
# ============================================================

def find_stock_csv():

    possible_files = [
        "Stock dataset (1).csv",
        "Stock dataset (2).csv",
        "Stock dataset.csv",
        "stock_dataset.csv",
    ]

    for filename in possible_files:
        if os.path.exists(filename):
            return filename

    return None


def find_rf_csv():

    possible_files = [
        "rf_feature_importance (1).csv",
        "RF feature importance.csv",
        "rf_feature_importance.csv",
        "rf_feature_importance (1).CSV",
    ]

    for filename in possible_files:
        if os.path.exists(filename):
            return filename

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_stock_data(filename):

    df = pd.read_csv(filename)

    # Remove accidental unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Clean column names
    df.columns = [
        str(c).strip().replace(" ", "_")
        for c in df.columns
    ]

    # Detect date column
    date_candidates = [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Timestamp",
        "timestamp"
    ]

    date_col = None

    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)

    return df


# ============================================================
# IDENTIFY STOCK/TICKER COLUMN
# ============================================================

def find_stock_column(df):

    candidates = [
        "Ticker",
        "ticker",
        "Symbol",
        "symbol",
        "Stock",
        "stock",
        "Company",
        "company"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


# ============================================================
# IDENTIFY DATE COLUMN
# ============================================================

def find_date_column(df):

    for col in [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Timestamp",
        "timestamp"
    ]:
        if col in df.columns:
            return col

    return None


# ============================================================
# PREPARE STOCK DATA
# ============================================================

def prepare_stock_data(df):

    df = df.copy()

    date_col = find_date_column(df)

    stock_col = find_stock_column(df)

    # Numeric conversion
    numeric_candidates = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return",
        "MA_5",
        "MA_10",
        "MA_20",
        "Volatility_20"
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # If Return does not exist, calculate it from Close
    if "Return" not in df.columns and "Close" in df.columns:

        if stock_col is not None:
            df["Return"] = (
                df.groupby(stock_col)["Close"]
                .pct_change()
            )
        else:
            df["Return"] = df["Close"].pct_change()

    # Create technical features if missing
    if "MA_5" not in df.columns and "Close" in df.columns:

        if stock_col is not None:
            df["MA_5"] = (
                df.groupby(stock_col)["Close"]
                .transform(lambda x: x.rolling(5).mean())
            )
        else:
            df["MA_5"] = df["Close"].rolling(5).mean()

    if "MA_10" not in df.columns and "Close" in df.columns:

        if stock_col is not None:
            df["MA_10"] = (
                df.groupby(stock_col)["Close"]
                .transform(lambda x: x.rolling(10).mean())
            )
        else:
            df["MA_10"] = df["Close"].rolling(10).mean()

    if "MA_20" not in df.columns and "Close" in df.columns:

        if stock_col is not None:
            df["MA_20"] = (
                df.groupby(stock_col)["Close"]
                .transform(lambda x: x.rolling(20).mean())
            )
        else:
            df["MA_20"] = df["Close"].rolling(20).mean()

    if "Volatility_20" not in df.columns and "Return" in df.columns:

        if stock_col is not None:
            df["Volatility_20"] = (
                df.groupby(stock_col)["Return"]
                .transform(lambda x: x.rolling(20).std())
            )
        else:
            df["Volatility_20"] = df["Return"].rolling(20).std()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# GET STOCK DATA
# ============================================================

def get_selected_stock(df, stock_name):

    stock_col = find_stock_column(df)

    if stock_col is None:
        return df.copy()

    selected = df[
        df[stock_col].astype(str) == str(stock_name)
    ].copy()

    return selected


# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_feature_columns(df):

    preferred_features = [
        "Return",
        "Volume",
        "MA_5",
        "MA_10",
        "MA_20",
        "Volatility_20",
        "Open",
        "High",
        "Low",
        "Close"
    ]

    features = [
        col
        for col in preferred_features
        if col in df.columns
    ]

    return features


# ============================================================
# LSTM MODEL
# ============================================================

class StockLSTM(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.10 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output).squeeze(1)


# ============================================================
# CREATE LSTM SEQUENCES
# ============================================================

def create_sequences(values, targets, window):

    X = []
    y = []

    for i in range(window, len(values)):

        X.append(
            values[i-window:i]
        )

        y.append(
            targets[i]
        )

    return np.array(X), np.array(y)


# ============================================================
# TRAIN LSTM
# ============================================================

def train_lstm(train_values, train_targets):

    X, y = create_sequences(
        train_values,
        train_targets,
        WINDOW
    )

    if len(X) < 30:
        return None

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    model = StockLSTM(
        input_size=train_values.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_function = nn.HuberLoss(
        delta=0.02
    )

    model.train()

    for epoch in range(100):

        for batch_x, batch_y in loader:

            optimizer.zero_grad()

            predictions = model(
                batch_x.to(DEVICE)
            )

            loss = loss_function(
                predictions,
                batch_y.to(DEVICE)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

    return model


# ============================================================
# LSTM PREDICTION
# ============================================================

def lstm_predict(model, X):

    model.eval()

    with torch.no_grad():

        tensor = torch.tensor(
            X,
            dtype=torch.float32
        ).to(DEVICE)

        predictions = model(
            tensor
        ).cpu().numpy()

    return predictions


# ============================================================
# TRAIN BOTH MODELS
# ============================================================

@st.cache_resource
def train_models(stock_name, csv_filename):

    df = load_stock_data(csv_filename)

    df = prepare_stock_data(df)

    df = get_selected_stock(
        df,
        stock_name
    )

    if len(df) < 100:
        raise ValueError(
            "Not enough data for this stock."
        )

    date_col = find_date_column(df)

    features = get_feature_columns(df)

    if "Return" not in features:

        raise ValueError(
            "The dataset needs a Return column "
            "or a Close column from which Return can be calculated."
        )

    if len(features) < 2:

        raise ValueError(
            "Not enough numerical features."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    if date_col:

        df = df.sort_values(
            date_col
        )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df_model = df[
        features
    ].copy()

    df_model = df_model.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(df_model) < 100:

        raise ValueError(
            "Not enough clean rows after preprocessing."
        )

    # --------------------------------------------------------
    # Target = next-day return
    # --------------------------------------------------------

    target = (
        df_model["Return"]
        .shift(-1)
    )

    valid = target.notna()

    df_model = df_model.loc[
        valid
    ]

    target = target.loc[
        valid
    ]

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    split = int(
        len(df_model) *
        (1 - TEST_RATIO)
    )

    X = df_model.values.astype(
        np.float32
    )

    y = target.values.astype(
        np.float32
    )

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_predictions = rf.predict(
        X_test
    )

    # ========================================================
    # LSTM
    # ========================================================

    # LSTM uses the same features and learns sequences.
    lstm_model = train_lstm(
        X_train,
        y_train
    )

    # Create test sequences using previous observations.
    combined_X = X[
        split - WINDOW:
    ]

    combined_y = y[
        split - WINDOW:
    ]

    lstm_X_test, lstm_y_test = create_sequences(
        combined_X,
        combined_y,
        WINDOW
    )

    lstm_predictions = lstm_predict(
        lstm_model,
        lstm_X_test
    )

    # --------------------------------------------------------
    # Dates for test data
    # --------------------------------------------------------

    if date_col:

        clean_dates = df.loc[
            df_model.index,
            date_col
        ].reset_index(
            drop=True
        )

        test_dates = clean_dates[
            split:
        ].reset_index(
            drop=True
        )

    else:

        test_dates = pd.RangeIndex(
            start=split,
            stop=len(df_model)
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def metrics(actual, prediction):

        r2 = r2_score(
            actual,
            prediction
        )

        mae = mean_absolute_error(
            actual,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                prediction
            )
        )

        direction = np.mean(
            np.sign(actual) ==
            np.sign(prediction)
        )

        return {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "Direction": direction
        }

    rf_metrics = metrics(
        y_test,
        rf_predictions
    )

    lstm_metrics = metrics(
        lstm_y_test,
        lstm_predictions
    )

    # --------------------------------------------------------
    # Recent prediction dataframe
    # --------------------------------------------------------

    n_recent = min(
        60,
        len(y_test),
        len(lstm_predictions)
    )

    recent_actual = y_test[
        -n_recent:
    ]

    recent_rf = rf_predictions[
        -n_recent:
    ]

    recent_lstm = lstm_predictions[
        -n_recent:
    ]

    recent_dates = list(
        test_dates[-n_recent:]
    )

    prediction_df = pd.DataFrame({

        "Date": recent_dates,

        "Actual": recent_actual,

        "Random Forest": recent_rf,

        "LSTM": recent_lstm
    })

    # --------------------------------------------------------
    # Next-day prediction
    # --------------------------------------------------------

    latest_features = X[
        -WINDOW:
    ]

    next_lstm = float(
        lstm_predict(
            lstm_model,
            latest_features.reshape(
                1,
                WINDOW,
                len(features)
            )
        )[0]
    )

    next_rf = float(
        rf.predict(
            X[-1].reshape(
                1,
                -1
            )
        )[0]
    )

    # LSTM is main model
    main_prediction = next_lstm

    direction_text = (
        "UP"
        if main_prediction >= 0
        else "DOWN"
    )

    # Simple directional confidence.
    # This is NOT a calibrated probability.
    historical_direction = np.mean(
        np.sign(
            lstm_y_test
        ) ==
        np.sign(
            lstm_predictions
        )
    )

    confidence = (
        50 +
        abs(
            historical_direction - 0.5
        ) * 100
    )

    confidence = float(
        np.clip(
            confidence,
            50,
            95
        )
    )

    # --------------------------------------------------------
    # Normalized close price
    # --------------------------------------------------------

    chart_df = df.copy()

    if "Close" in chart_df.columns:

        chart_df["Normalized_Close"] = (
            chart_df["Close"] /
            chart_df["Close"].iloc[0]
        )

    # --------------------------------------------------------
    # Return model results
    # --------------------------------------------------------

    model_results = pd.DataFrame({

        "Model": [
            "LSTM",
            "Random Forest"
        ],

        "R²": [
            lstm_metrics["R2"],
            rf_metrics["R2"]
        ],

        "MAE": [
            lstm_metrics["MAE"],
            rf_metrics["MAE"]
        ],

        "RMSE": [
            lstm_metrics["RMSE"],
            rf_metrics["RMSE"]
        ],

        "Direction Accuracy": [
            lstm_metrics["Direction"],
            rf_metrics["Direction"]
        ]
    })

    return {
        "df": df,
        "features": features,
        "rf": rf,
        "lstm": lstm_model,
        "model_results": model_results,
        "prediction_df": prediction_df,
        "next_lstm": next_lstm,
        "next_rf": next_rf,
        "direction": direction_text,
        "confidence": confidence,
        "chart_df": chart_df
    }


# ============================================================
# LOAD RF FEATURE IMPORTANCE FILE
# ============================================================

@st.cache_data
def load_rf_importance(filename):

    if filename is None:
        return None

    try:

        df = pd.read_csv(
            filename
        )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        if "Feature" not in df.columns:
            return None

        if "Importance" not in df.columns:
            return None

        df["Importance"] = pd.to_numeric(
            df["Importance"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Importance"]
        )

        return df.sort_values(
            "Importance",
            ascending=True
        )

    except Exception:
        return None


# ============================================================
# SIDEBAR
# ============================================================

stock_csv = find_stock_csv()

if stock_csv is None:

    st.error(
        "Stock dataset CSV not found beside app.py."
    )

    st.info(
        "Put your Stock dataset CSV in the same "
        "GitHub folder as app.py."
    )

    st.stop()


df_raw = load_stock_data(
    stock_csv
)

df_raw = prepare_stock_data(
    df_raw
)

stock_column = find_stock_column(
    df_raw
)

if stock_column:

    stocks = sorted(
        df_raw[
            stock_column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    stocks = ["Stock"]


with st.sidebar:

    st.markdown(
        "## 📊 AI Stock Prediction"
    )

    st.markdown(
        "---"
    )

    selected_stock = st.selectbox(
        "Select Stock",
        stocks
    )

    st.markdown(
        "### Date Range"
    )

    date_column = find_date_column(
        df_raw
    )

    if date_column:

        dates = pd.to_datetime(
            df_raw[date_column],
            errors="coerce"
        ).dropna()

        if len(dates):

            st.date_input(
                "Available data",
                value=(
                    dates.min().date(),
                    dates.max().date()
                ),
                disabled=True
            )

    st.markdown("---")

    st.caption(
        "Main model: LSTM"
    )

    st.caption(
        "Comparison model: Random Forest"
    )

    st.caption(
        "Target: next-day stock return"
    )


# ============================================================
# TRAIN
# ============================================================

try:

    with st.spinner(
        "Training LSTM and Random Forest models..."
    ):

        results = train_models(
            selected_stock,
            stock_csv
        )

except Exception as e:

    st.error(
        f"Model error: {e}"
    )

    st.info(
        "Check that your CSV contains Date, "
        "Close and preferably Return, Volume, "
        "Open, High and Low columns."
    )

    st.stop()


model_results = results[
    "model_results"
]

prediction_df = results[
    "prediction_df"
]

chart_df = results[
    "chart_df"
]

features = results[
    "features"
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📈 AI Stock Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">LSTM Main Model vs Random Forest</div>',
    unsafe_allow_html=True
)


# ============================================================
# BEST MODEL METRICS
# ============================================================

best_lstm = model_results[
    model_results["Model"] == "LSTM"
].iloc[0]

best_rf = model_results[
    model_results["Model"] == "Random Forest"
].iloc[0]

# Main model is LSTM
best_model = best_lstm
best_model_name = "LSTM"


def metric_card(
    title,
    value,
    model
):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-model">{model}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


c1, c2, c3, c4 = st.columns(4)

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "R² Score",
        f"{float(best_model['R²']):.4f}",
        best_model_name
    )

with c2:
    metric_card(
        "MAE",
        f"{float(best_model['MAE']):.4f}",
        best_model_name
    )

with c3:
    metric_card(
        "RMSE",
        f"{float(best_model['RMSE']):.4f}",
        best_model_name
    )

with c4:
    metric_card(
        "Direction Accuracy",
        f"{float(best_model['Direction Accuracy']) * 100:.2f}%",
        best_model_name
    )


st.markdown("")


# ============================================================
# PRICE + MODEL COMPARISON
# ============================================================

left, right = st.columns(
    [1.35, 1]
)


with left:

    st.subheader(
        "Stock Close Price"
    )

    if "Close" in chart_df.columns:

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        if "Date" in chart_df.columns:

            ax.plot(
                chart_df["Date"],
                chart_df["Close"],
                linewidth=1.2
            )

            ax.set_xlabel(
                "Date"
            )

        else:

            ax.plot(
                chart_df["Close"],
                linewidth=1.2
            )

        ax.set_ylabel(
            "Close Price"
        )

        ax.grid(
            alpha=0.20
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )


with right:

    st.subheader(
        "Model Comparison — R²"
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    colors = [
        "#3b82f6",
        "#f59e0b"
    ]

    bars = ax.bar(
        model_results["Model"],
        model_results["R²"],
        color=colors
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_ylabel(
        "R² Score"
    )

    for bar, value in zip(
        bars,
        model_results["R²"]
    ):

        ax.text(
            bar.get_x() +
            bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top"
        )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# SECOND ROW
# ============================================================

left2, mid2, right2 = st.columns(
    [1, 1.25, 1]
)


# ============================================================
# DIRECTION ACCURACY
# ============================================================

with left2:

    st.subheader(
        "Direction Accuracy"
    )

    names = model_results[
        "Model"
    ].tolist()

    values = (
        model_results[
            "Direction Accuracy"
        ] * 100
    ).tolist()

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    bars = ax.bar(
        names,
        values,
        color=[
            "#22c55e",
            "#f59e0b"
        ]
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_ylim(
        0,
        100
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x() +
            bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center"
        )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

with mid2:

    st.subheader(
        "Feature Importance — Random Forest"
    )

    rf_importance_file = find_rf_csv()

    rf_importance = load_rf_importance(
        rf_importance_file
    )

    if rf_importance is not None:

        top_features = rf_importance.tail(
            min(10, len(rf_importance))
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.grid(
            axis="x",
            alpha=0.20
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    else:

        # Use trained RF importance
        importance = pd.DataFrame({

            "Feature": features,

            "Importance": results[
                "rf"
            ].feature_importances_
        }).sort_values(
            "Importance"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.barh(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_xlabel(
            "Importance"
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )


# ============================================================
# NEXT DAY PREDICTION
# ============================================================

with right2:

    st.subheader(
        "Next Day Prediction"
    )

    next_prediction = results[
        "next_lstm"
    ]

    direction = results[
        "direction"
    ]

    confidence = results[
        "confidence"
    ]

    if direction == "UP":

        prediction_html = (
            '<div class="prediction-up">↑ UP</div>'
        )

    else:

        prediction_html = (
            '<div class="prediction-down">↓ DOWN</div>'
        )

    st.markdown(
        f"""
        <div class="prediction-box">

            <div>
                Predicted Direction
            </div>

            {prediction_html}

            <hr>

            <div>
                LSTM Confidence Indicator
            </div>

            <h2>
                {confidence:.2f}%
            </h2>

            <div>
                Predicted Return
            </div>

            <h2>
                {next_prediction:.4f}
                ({next_prediction * 100:.2f}%)
            </h2>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Confidence is a directional indicator based on "
        "historical test accuracy; it is not a calibrated "
        "probability."
    )


# ============================================================
# RECENT PREDICTIONS
# ============================================================

st.subheader(
    "Recent Predictions vs Actual Returns"
)

fig, ax = plt.subplots(
    figsize=(15, 5)
)

if "Date" in prediction_df.columns:

    ax.plot(
        prediction_df["Date"],
        prediction_df["Actual"],
        label="Actual Return",
        linewidth=1.5
    )

    ax.plot(
        prediction_df["Date"],
        prediction_df["LSTM"],
        label="Predicted Return (LSTM)",
        linewidth=1.5
    )

    ax.plot(
        prediction_df["Date"],
        prediction_df["Random Forest"],
        label="Predicted Return (RF)",
        linestyle="--",
        linewidth=1.3
    )

else:

    ax.plot(
        prediction_df["Actual"],
        label="Actual Return"
    )

    ax.plot(
        prediction_df["LSTM"],
        label="Predicted Return (LSTM)"
    )

    ax.plot(
        prediction_df["Random Forest"],
        label="Predicted Return (RF)",
        linestyle="--"
    )

ax.axhline(
    0,
    linewidth=1
)

ax.set_ylabel(
    "Return"
)

ax.set_xlabel(
    "Date"
)

ax.legend()

ax.grid(
    alpha=0.20
)

fig.tight_layout()

st.pyplot(
    fig,
    clear_figure=True
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

st.subheader(
    "Detailed Model Performance"
)

display_results = model_results.copy()

display_results[
    "Direction Accuracy"
] = (
    display_results[
        "Direction Accuracy"
    ] * 100
).round(2).astype(str) + "%"

display_results = display_results.rename(
    columns={
        "R2": "R²",
        "Direction Accuracy":
            "Direction Accuracy"
    }
)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ABOUT
# ============================================================

st.subheader(
    "About This Project"
)

st.write(
    """
    This AI stock prediction system uses a Long Short-Term Memory
    (LSTM) neural network as the main prediction model and Random
    Forest as a comparison model.

    The system learns from historical stock-market data and predicts
    the next-day stock return.

    The LSTM receives sequences of historical market observations,
    allowing it to learn temporal patterns. Random Forest uses
    engineered market features and provides feature-importance
    information.

    The system evaluates the models using R², MAE, RMSE and
    directional accuracy.
    """
)

st.info(
    "Important: stock-return prediction is experimental. "
    "A high dashboard score does not guarantee future profits."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "AI Stock Prediction App | Built with Streamlit | "
    "LSTM + Random Forest"
)
