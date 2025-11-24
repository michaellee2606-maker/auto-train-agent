from smolagents import CodeAgent, InferenceClientModel

class MachineLearningAgent:
    def __init__(self, model_id, token):
        self.model = InferenceClientModel(model_id=model_id, token=token)
        self.agent = CodeAgent(tools=[], model=self.model, additional_authorized_imports=["pandas","h2o.automl"])
    
    def train(self):
        self.agent.run("machine learning")