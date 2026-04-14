import io
import base64
import logging
import numpy as np
import matplotlib.pyplot as plt
from responseModels.FeatureSelectionResponse import FeatureSelectionResponse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, feature_selection_analysis_model_id, feature_selection_format_model_id, token, 
                 feature_type, score_func, reports_dict):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.feature_selection_analysis_model_id = feature_selection_analysis_model_id
        self.feature_selection_format_model_id = feature_selection_format_model_id
        self.token = token

        self.feature_type = feature_type
        self.score_func = score_func
        self.reports_dict = reports_dict
        self.feature_selector = SelectPercentile(score_func)
        self.input_features = None  # To store input feature names
        self.index_list_of_selected_features = []

        self.langfuse_handler = CallbackHandler()
        self.feature_selection_analysis_agent = self.init_feature_selection_analysis_agent()
        self.feature_selection_format_agent = self.init_feature_selection_format_agent()

    def init_feature_selection_analysis_agent(self):
        feature_selection_analysis_llm = HuggingFaceEndpoint(
            task='conversational',
            repo_id=self.feature_selection_analysis_model_id,
            temperature=0.2,
            huggingfacehub_api_token=self.token,
            provider="auto",
        )

        feature_selection_analysis_chat = ChatHuggingFace(llm=feature_selection_analysis_llm)

        feature_selection_analysis_agent = create_agent(
            model=feature_selection_analysis_chat,
            tools=[],
            system_prompt="""
                You are a professional data analyst.
                The business problem is to predict the class of customers based on the customer data. This is a binary classification problem. 
            """
        )

        return feature_selection_analysis_agent

    def init_feature_selection_format_agent(self):
        llm = HuggingFaceEndpoint(
            task='conversational',
            repo_id=self.feature_selection_format_model_id,
            temperature=0.3,
            huggingfacehub_api_token=self.token,
            provider="auto",
        )

        chat = ChatHuggingFace(llm=llm)

        agent = create_agent(
            model=chat,
            tools=[],
            system_prompt="""
                You are a professional data analyst.
                The business problem is to predict the class of customers based on the customer data. This is a binary classification problem. 
            """,
            response_format=FeatureSelectionResponse
        )

        return agent

    def fit(self, X, y):
        self.feature_selector.fit_transform(X, y)
        self.input_features = X.columns if hasattr(X, 'columns') else np.arange(X.shape[-1])
        
        feature_scores = -np.log10(self.feature_selector.pvalues_)
        feature_scores /= feature_scores.max()

        buf = io.BytesIO()

        feature_indices = np.arange(X.shape[-1])
        plt.bar(feature_indices, feature_scores)
        plt.title(f"{self.feature_type.capitalize()} Feature Univariate Score")
        plt.xlabel("Feature number")
        plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
        plt.savefig(buf, format="jpeg")
        plt.close()

        # Retrieve the byte data from the stream and encode it to base64
        # Use .decode('utf-8') to convert the bytes object into a standard string
        base64_feature_univariate_score = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        self.reports_dict[f"{self.feature_type.capitalize()} Feature Univariate Score"] = base64_feature_univariate_score

        # From base64 data
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64, {base64_feature_univariate_score}"},
                },
                {"type": "text", "text": "according to the picture, give me a reasonable number of features chosen. The result which is returned is an array of index of features chosen."}
            ]
        }

        feature_selection_analysis_result = self.feature_selection_analysis_agent.invoke(
            {"messages": [message]}, 
            config={"callbacks": [self.langfuse_handler]}
        )
        
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": feature_selection_analysis_result['messages'][-1].content}
            ]
        }

        response = self.feature_selection_format_agent.invoke(
            {"messages": [message]}, 
            config={"callbacks": [self.langfuse_handler]}
        )
        
        self.logger.info(f"{self.feature_type.capitalize()} features selection result is : {response['structured_response']}")

        self.index_list_of_selected_features = response['structured_response'].chosen_features
        
        return self

    def transform(self, X):
        index_list_of_deleted_features = []

        for index in range(X.shape[-1]):
            if index not in self.index_list_of_selected_features:
                index_list_of_deleted_features.append(index)

        return np.delete(X, index_list_of_deleted_features, 1)
    
    def get_feature_names_out(self, input_features=None):
        # Use input_features if provided, otherwise use stored input features
        if input_features is None:
            input_features = self.input_features

        # Return the selected feature names
        features_chosen = [input_features[i] for i in self.index_list_of_selected_features]

        self.logger.info(f"{self.feature_type.capitalize()} features chosen are : {features_chosen}")

        return features_chosen