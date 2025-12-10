import h2o
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from agents.DataAnalysisAgent import DataAnalysisAgent
from agents.MachineLearningAgent import MachineLearningAgent

class AutoTrain:
    def __init__(self, model_id, token, font_path):
        register()
        SmolagentsInstrumentor().instrument()
        
        h2o.init()

        self.dataAnalysisAgent = DataAnalysisAgent(model_id, token)
        self.machineLearningAgent = MachineLearningAgent(model_id, token, font_path)

    def start(self, train_data_path: str, validate_data_path: str, out_directory: str):
        # self.dataAnalysisAgent.analyze()
        self.machineLearningAgent.train(train_data_path, out_directory)
        self.machineLearningAgent.generate_report(validate_data_path, out_directory)
   