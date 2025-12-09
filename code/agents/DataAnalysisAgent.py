from smolagents import CodeAgent, InferenceClientModel

class DataAnalysisAgent:
    def __init__(self, model_id, token):
        self.model = InferenceClientModel(model_id=model_id, token=token)
        self.agent = CodeAgent(tools=[], model=self.model, additional_authorized_imports=["pandas","h2o.automl"])
    
    def analyze(self):
        self.agent.run("feature engineering and data analysis")