import h2o
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from agents.DataAnalysisAgent import DataAnalysisAgent
from agents.MachineLearningAgent import MachineLearningAgent

class AutoTrain:
    def __init__(self, model_id, token, font_path, class_column, postive_class, negative_class):
        # Initialize telemetry instrumentation
        register()
        SmolagentsInstrumentor().instrument()
        
        # Initialize H2O
        h2o.init()

        self.dataAnalysisAgent = DataAnalysisAgent(model_id, token, class_column, postive_class, negative_class)
        self.machineLearningAgent = MachineLearningAgent(model_id, token, font_path, class_column)

    def start(self, train_data_path: str, validate_data_path: str, out_directory: str):
        # Data analysis and feature extraction
        train_feature_path, validate_feature_path = self.dataAnalysisAgent.analyze(train_data_path, validate_data_path)
        # Machine learning model training
        self.machineLearningAgent.train(train_feature_path, out_directory)
        # Generate report
        self.machineLearningAgent.generate_report(validate_feature_path, out_directory)
   