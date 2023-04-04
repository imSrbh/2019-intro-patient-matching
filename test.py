import gradio as gr
import pandas as pd

def analyze_csv(file):
    # Read the uploaded CSV file
    df = pd.read_csv(file)
    
    # Perform some analysis on the data (for example)
    num_rows = len(df)
    num_cols = len(df.columns)
    summary = df.describe()
    
    # Return the analysis results
    return f"Number of Rows: {num_rows}\nNumber of Columns: {num_cols}\n\nData Summary:\n{summary}"

# Create the Gradio interface
inputs = gr.inputs.File(label="Upload CSV")
outputs = gr.outputs.Textbox(label="CSV Analysis")
app = gr.Interface(analyze_csv, inputs, outputs, title="CSV Analyzer")
app.launch()
