from agents.DataAnalysisAgent import DataAnalysisAgent
from agents.MachineLearningAgent import MachineLearningAgent

class AutoTrain:
    def __init__(self, model_id, token):
        self.dataAnalysisAgent = DataAnalysisAgent(model_id, token)
        self.machineLearningAgent = MachineLearningAgent(model_id, token)

    def start(self, data_path: str, out_directory: str):
        # self.dataAnalysisAgent.analyze()
        self.machineLearningAgent.train(data_path, out_directory)
   