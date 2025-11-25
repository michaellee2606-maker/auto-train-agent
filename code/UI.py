import os
from AutoTrain import AutoTrain
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

model_id = os.environ.get("model_id")
token = os.environ.get("token")
out_directory = os.environ.get("out_directory")


def generate(file):
    if file is None:
        raise gr.Error("No file uploaded!")

    autoTrain = AutoTrain(model_id, token)

    autoTrain.start(file.name)
    
    files = [out_directory+"\\"+file 
                for file in os.listdir(out_directory) 
                    if os.path.isfile(os.path.join(out_directory, file))]

    return gr.File(value=files)


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
    generate_button.click(fn=generate, inputs=input_file, outputs=output_file)
    

demo.launch(allowed_paths=[out_directory])