import os
import h2o
import logging
from h2o.estimators import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from responseModels.XGBoostResponse import XGBoostResponse


class MachineLearningAgent:
    def __init__(self, model_id, token, font_path, class_column):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.fontProps = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = self.fontProps.get_name()
        self.class_column = class_column
        self.best_xgboost_model = None
        self.max_f2_score = 0
        self.threshold_of_best_model = 0
        self.confusion_matrix = None
        self.recall = 0
        self.precision = 0
        self.langfuse_handler = CallbackHandler()
        self.agent = self.init_agent(model_id, token)

    def init_agent(self, model_id, token):
        llm = HuggingFaceEndpoint(
            task='conversational',
            repo_id=model_id,
            max_new_tokens=1280,
            temperature=0.5,
            huggingfacehub_api_token=token,
            provider="auto",
        )

        chat = ChatHuggingFace(llm=llm)

        agent = create_agent(
            model=chat,
            tools=[],
            system_prompt="""
                You are a professional machine learning assistant.
                The business problem is to predict the class of customers based on the customer data. This is a binary classification problem. 
                The response should conform to the corresponding response format.
            """,
            response_format=XGBoostResponse
        )

        return agent

    
    def train(self, train_feature_path: str, validate_feature_path: str, out_directory: str):
        train_features = h2o.import_file(train_feature_path)
        train_features[self.class_column] = train_features[self.class_column].asfactor()
        validate_features = h2o.import_file(validate_feature_path)
        actual = validate_features.pop(self.class_column).asfactor()

        # Run the agent
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": "Generate the hyperparameters and search criteria of XGBoost algorithms of H2OGridSearch API."}]}, 
            config={"callbacks": [self.langfuse_handler]}
        )

        xgboost_response_json = XGBoostResponse.model_dump(response["structured_response"])

        xgboost_params = xgboost_response_json.get("hyperparameters")
        self.logger.info(f"XGBoost hyperparameters: {xgboost_params}")

        search_criteria = xgboost_response_json.get("search_criteria")
        self.logger.info(f"XGBoost search criteria: {search_criteria}")

        xgboost_grid = H2OGridSearch(
            grid_id='xgboost_grid',
            model=H2OXGBoostEstimator,
            hyper_params=xgboost_params,
            search_criteria=search_criteria,
            parallelism=0
        )

        xgboost_grid.train(
            y=self.class_column,
            training_frame=train_features,
            nfolds=5,
            fold_assignment='Stratified',
        )

        xgboost_models = xgboost_grid.get_grid()

        for xgboost_model in xgboost_models:
            threshold = xgboost_model.F2()[0][0]

            validate_result = xgboost_model.predict(validate_features)
            validate_result['predict'] = (validate_result['p1']>=threshold).ifelse(1, 0)

            combined_result = validate_result['predict'].cbind(actual)
            grouped_result = combined_result.group_by(['predict', 'class']).count().get_frame()
            pivot_result = grouped_result.pivot(index='class', column='predict', value='nrow')
            confusion_matrix = pivot_result.as_data_frame(use_multi_thread=True, use_pandas=True, header=True)

            # Calculate Precision and Recall
            precision = confusion_matrix.iloc[1, 2] / (confusion_matrix.iloc[0, 2] + confusion_matrix.iloc[1, 2])
            recall = confusion_matrix.iloc[1, 2] / (confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 2])

            # Calculate F2 Score
            f2_score = (5 * precision * recall) / (4 * precision + recall)

            self.logger.info(f"Model: {xgboost_model.model_id}, Threshold: {threshold}, Precision: {precision:.4f}, Recall: {recall:.4f}, F2 Score: {f2_score:.4f}")

            if f2_score > self.max_f2_score:
                self.max_f2_score = f2_score 
                self.best_xgboost_model = xgboost_model
                self.threshold_of_best_model = threshold
                self.confusion_matrix = confusion_matrix
                self.recall = recall
                self.precision = precision

        self.best_xgboost_model.download_mojo(path=out_directory, get_genmodel_jar=True)

    def generate_report(self, out_directory: str, reports_dict):
        confusion_matrix = self.confusion_matrix
        model = self.best_xgboost_model.model_id

        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=confusion_matrix.values, colLabels=confusion_matrix.columns, 
            loc='center', colLoc='center', cellLoc='center')

        # Add a title for the table
        plt.title('混淆矩阵', fontsize=14, fontweight='bold', fontproperties=self.fontProps)

        # Calculate Precision and Recall
        precision = confusion_matrix.iloc[1, 2] / (confusion_matrix.iloc[0, 2] + confusion_matrix.iloc[1, 2])
        recall = confusion_matrix.iloc[1, 2] / (confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 2])

        # Add Precision and Recall to the plot
        plt.figtext(0.5, 0.01, f'召回率: {recall:.2f}, 精确率: {precision:.2f}', wrap=True, horizontalalignment='center', fontsize=12, fontproperties=self.fontProps)

        # Save the table as a PDF
        report_path = out_directory + os.sep + f'Report - {model}.pdf'
        plt.savefig(report_path)