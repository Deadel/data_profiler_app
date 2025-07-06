import gradio as gr
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from utils import load_data, describe_data, plot_missing_data, correlation_heatmap, generate_plot, run_simple_model
from PIL import Image

state = {"df": None}

def analyze_file(file):
    if file is None:
        return "Brak pliku", "", "", None, gr.update(choices=[]), gr.update(choices=[])

    # file może być pathlib.Path lub TemporaryFileWrapper lub file-like object
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Błąd wczytywania pliku: {str(e)}", "", "", None, gr.update(choices=[]), gr.update(choices=[])

    state["df"] = df

    preview = df.head().to_markdown()
    stats = describe_data(df).to_markdown()
    missing = plot_missing_data(df)
    missing_text = missing.to_markdown() if not missing.empty else "Brak braków danych!"

    heatmap_buf = correlation_heatmap(df)
    heatmap_img = Image.open(heatmap_buf) if heatmap_buf else None

    return preview, stats, missing_text, heatmap_img, gr.update(choices=df.columns.tolist()), gr.update(choices=df.columns.tolist())

def plot_selected_column(column):
    df = state.get("df")
    if df is not None and column in df.columns:
        fig = generate_plot(df, column)
        return fig
    return px.scatter(x=[], y=[])

def run_model(target):
    df = state.get("df")
    if df is not None and target:
        result, model_path = run_simple_model(df.copy(), target)
        if model_path:
            return f"✅ Model zapisany do: {model_path}\n\n{result}"
        return result
    return "Brak danych lub niewybrana kolumna docelowa."

with gr.Blocks() as demo:
    gr.Markdown("# 📊 Data Profiler - Szybka analiza danych CSV")

    with gr.Row():
        file_input = gr.File(label="Wgraj plik CSV")
        column_selector = gr.Dropdown(label="Wybierz kolumnę", choices=[])

    with gr.Row():
        preview_out = gr.Markdown()
        stats_out = gr.Markdown()

    with gr.Row():
        missing_out = gr.Markdown()
        heatmap_out = gr.Image(type="pil")

    with gr.Row():
        plot_out = gr.Plot()

    with gr.Row():
        gr.Markdown("## 🧠 Prosty model ML (Random Forest Classifier)")
        model_target = gr.Dropdown(label="Kolumna docelowa", choices=[])
        model_result = gr.Textbox(label="Wynik klasyfikatora")
        model_button = gr.Button("Trenuj model")

    file_input.change(fn=analyze_file, inputs=file_input,
                      outputs=[preview_out, stats_out, missing_out, heatmap_out, column_selector, model_target])
    column_selector.change(fn=plot_selected_column, inputs=column_selector, outputs=plot_out)
    model_button.click(fn=run_model, inputs=model_target, outputs=model_result)

if __name__ == "__main__":
    demo.launch()
