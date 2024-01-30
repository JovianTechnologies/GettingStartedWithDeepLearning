import gradio as gr
import main
import os

model_list = ['models/' + model for model in os.listdir('models')]

demo = gr.Interface(
    title="Chanterelle or Not",
    fn=main.predict_image,
    inputs=[gr.Dropdown(model_list), "image"],
    outputs=["text"],
)

demo.launch()