import gradio as gr
from utils import process_file

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AutoTrain Agent
        <br>
        Upload the data file, then click the Generate button, you will get a model!
        """
    )
    input_file = gr.File(label="Data File", file_count="single")
    generate_button = gr.Button("Generate")
    output_file = gr.File(label="Generated Files", file_count="multiple")
    generate_button.click(fn=process_file, inputs=input_file, outputs=output_file)

demo.launch() 