import h2o
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from agents.DataAnalysisAgent import DataAnalysisAgent
from agents.MachineLearningAgent import MachineLearningAgent

class AutoTrain:
    def __init__(self, model_id, token):
        register()
        SmolagentsInstrumentor().instrument()
        
        h2o.init()

        self.dataAnalysisAgent = DataAnalysisAgent(model_id, token)
        self.machineLearningAgent = MachineLearningAgent(model_id, token)

    def start(self, train_data_path: str, validate_data_path: str, out_directory: str):
        train_feature_path, validate_feature_path = self.dataAnalysisAgent.analyze(train_data_path, validate_data_path)
        self.machineLearningAgent.train(train_feature_path, out_directory)
        self.machineLearningAgent.generate_report(validate_feature_path, out_directory)
   