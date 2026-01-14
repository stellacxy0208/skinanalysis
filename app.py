import os
import tempfile
import gradio as gr
from visia_filter.core import run_analysis

def analyze(image_path):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "analysis_panel.png")
        run_analysis(image_path, out, max_width=1100)
        return out

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="filepath", label="Upload a face photo"),
    outputs=gr.Image(type="filepath", label="VISIA-style panel (approx)"),
    title="VISIA-style Face Analysis (Python)",
    description="Approximation from standard photos. Real UV/porphyrin requires special imaging hardware."
)

if __name__ == "__main__":
    demo.launch()
