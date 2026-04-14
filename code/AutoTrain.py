import h2o
from agents.DataAnalysisAgent import DataAnalysisAgent
from agents.MachineLearningAgent import MachineLearningAgent
from agents.ReportAnalysisAgent import ReportAnalysisAgent

class AutoTrain:
    def __init__(self, model_id, multimodel_model_id, token, font_path, class_column, positive_class, negative_class):        
        # Initialize H2O
        h2o.init()

        self.dataAnalysisAgent = DataAnalysisAgent(multimodel_model_id, model_id, token, class_column, positive_class, negative_class)
        self.machineLearningAgent = MachineLearningAgent(model_id, token, class_column)
        self.reportAnalysisAgent = ReportAnalysisAgent(multimodel_model_id, token, font_path)

    def start(self, train_data_path: str, validate_data_path: str, out_directory: str):
        # Data analysis and feature extraction
        train_feature_path, validate_feature_path, reports_dict = self.dataAnalysisAgent.analyze(train_data_path, validate_data_path)
        # Machine learning model training
        model_id_trained, confusion_matrix, xgboost_response_json = self.machineLearningAgent.train(train_feature_path, validate_feature_path, out_directory)
        # Generate report
        report_analysis_result = self.reportAnalysisAgent.generate_report(model_id_trained, confusion_matrix, xgboost_response_json, out_directory, reports_dict)