import os
from AutoTrain import AutoTrain
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

model_id = os.environ.get("model_id")
token = os.environ.get("token")
out_directory = "." + os.sep + os.environ.get("out_directory")
font_path = os.environ.get("font_path")
class_column= os.environ.get("class_column")
positive_class= os.environ.get("positive_class")
negative_class= os.environ.get("negative_class")

# Initialize AutoTrain agent
autoTrain = AutoTrain(model_id, token, font_path, class_column, positive_class, negative_class)

def generate(train_data, validate_data):
    if train_data is None:
        raise gr.Error("请上传训练数据集!")
    
    if validate_data is None:
        raise gr.Error("请上传验证数据集!")

    # Start training process
    autoTrain.start(train_data.name, validate_data.name, out_directory)
    
    # Collect output files
    files = [out_directory + os.sep + file 
                for file in os.listdir(out_directory) 
                    if os.path.isfile(os.path.join(out_directory, file))]

    return gr.File(value=files)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 自动训练智能体
        <br>
        请上传训练数据集和验证数据集，然后点击生成模型按钮，最终会生成模型和报告等相关文件!
        """
    )
    train_data = gr.File(label="训练数据集", file_count="single")
    validate_data = gr.File(label="验证数据集", file_count="single")
    generate_button = gr.Button("生成模型")
    output_file = gr.File(label="生成的文件", file_count="multiple")
    generate_button.click(fn=generate, inputs=[train_data, validate_data], outputs=output_file)
    

#demo.launch(allowed_paths=[out_directory])

demo.launch(
    allowed_paths=[out_directory],
    server_name="127.0.0.1",  # 绑定所有网卡，而非仅127.0.0.1
    server_port=7860        # 端口保持7860（可自定义，如8080）
)