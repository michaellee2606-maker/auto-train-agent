import os
import gradio as gr
import autoTrainAgent as agent

def list_files_in_directory(directory):
    try:
        files = os.listdir(directory)
        return [directory+"\\"+file for file in files if os.path.isfile(os.path.join(directory, file))]
    except Exception as e:
        return str(e)


def process_file(file):
    if file is None:
        return "No file uploaded."
    
    agent.generate_model(file.name)

    files = list_files_in_directory(".\\generatedFiles")

    return gr.File(value=files)