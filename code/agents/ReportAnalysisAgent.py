import os
import io
import pandas as pd
import base64
import logging
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


class ReportAnalysisAgent:
    def __init__(self, model_id, token, font_path):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.fontProps = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = self.fontProps.get_name()
        self.langfuse_handler = CallbackHandler()
        self.agent = self.init_agent(model_id, token)

    def init_agent(self, model_id, token):
        llm = HuggingFaceEndpoint(
            task='conversational',
            repo_id=model_id,
            max_new_tokens=12800,
            temperature=0.5,
            huggingfacehub_api_token=token,
            provider="auto",
        )

        chat = ChatHuggingFace(llm=llm)

        agent = create_agent(
            model=chat,
            tools=[],
            system_prompt="""
                You are a professional report analysis assistant. 
                Proficient at analyse confusion metrics, feature importance grid, hyperparameters and search criteria of XGBoost algorithms of H2OGridSearch API.
                The business problem is to predict the class of customers based on the customer data. This is a binary classification problem. 
            """
        )

        return agent

    def generate_report(self, model_id_trained:str, confusion_matrix:pd.DataFrame, xgboost_response_json:str, out_directory:str, reports_dict):
        # Create a figure with subplots to accommodate the confusion matrix and images
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows: confusion matrix + 2 images
        axs[0].axis('tight')
        axs[0].axis('off')
        axs[0].table(cellText=confusion_matrix.values, colLabels=confusion_matrix.columns, 
                     loc='center', colLoc='center', cellLoc='center')

        # Add a title for the table
        axs[0].set_title('混淆矩阵', fontsize=14, fontweight='bold', fontproperties=self.fontProps)

        # Calculate Precision and Recall
        precision = confusion_matrix.iloc[1, 2] / (confusion_matrix.iloc[0, 2] + confusion_matrix.iloc[1, 2])
        recall = confusion_matrix.iloc[1, 2] / (confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 2])

        # Adjust the position of the text to be closer to the confusion matrix
        fig.text(0.5, 0.7, f'召回率: {recall:.2f}, 精确率: {precision:.2f}', wrap=True, horizontalalignment='center', fontsize=8, fontproperties=self.fontProps)

        # Decode and add the PNG images from reports_dict
        if 'Numerical Feature Univariate Score' in reports_dict:
            numerical_image_data = base64.b64decode(reports_dict['Numerical Feature Univariate Score'])
            numerical_image = Image.open(BytesIO(numerical_image_data))
            axs[1].imshow(numerical_image)
            axs[1].axis('off')  # Hide axes for the image
            axs[1].set_title("数值特征单变量分数", fontsize=14, fontproperties=self.fontProps)
        if 'Categorical Feature Univariate Score' in reports_dict:
            categorical_image_data = base64.b64decode(reports_dict['Categorical Feature Univariate Score'])
            categorical_image = Image.open(BytesIO(categorical_image_data))
            axs[2].imshow(categorical_image)
            axs[2].axis('off')  # Hide axes for the image
            axs[2].set_title("类别特征单变量分数", fontsize=14, fontproperties=self.fontProps)

        # Save the complete report as a PDF
        report_path = out_directory + os.sep + f'Report - {model_id_trained}.pdf'
        plt.tight_layout()
        plt.savefig(report_path)

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        plt.close()

        # Retrieve the byte data from the stream and encode it to base64
        # Use .decode('utf-8') to convert the bytes object into a standard string
        base64_report = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # From base64 data
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64, {base64_report}"},
                },
                {"type": "text", "text": f"According to the report as shown in the picture and the parameters of H2OGridSearch API which is {xgboost_response_json}, give me some reasonable advices of feature engineering and machine learning to improve the recall and precision metrics in the next iteration"}
            ]
        }

        report_analysis_result = self.agent.invoke(
            {"messages": [message]}, 
            config={"callbacks": [self.langfuse_handler]}
        )

        self.logger.info(f"Report analysis result is : {report_analysis_result['messages'][-1].content}")

        return report_analysis_result['messages'][-1].content