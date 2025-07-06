import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
from datetime import datetime

def load_data(file):
    return pd.read_csv(file)

def describe_data(df):
    return df.describe(include='all')

def plot_missing_data(df):
    missing = df.isnull().sum()
    return missing[missing > 0]

def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def generate_plot(df, column):
    if df[column].dtype == 'object':
        fig = px.histogram(df, x=column)
    else:
        fig = px.box(df, y=column)
    return fig

def run_simple_model(df, target_col):
    df = df.dropna()
    df = pd.get_dummies(df)

    if df.empty or target_col not in df.columns:
        return "Brak wystarczających danych do trenowania modelu.", None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if len(X) == 0 or len(y) == 0:
        return "Zbiór danych po czyszczeniu jest pusty.", None

    if len(X) < 5:
        return f"Zbyt mały zbiór do trenowania modelu: {len(X)} próbek.", None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)

        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/model_{timestamp}.pkl"
        dump(model, model_path)

        return report, model_path

    except Exception as e:
        return f"Model training error: {str(e)}", None
