from smolagents import CodeAgent, InferenceClientModel

class MachineLearningAgent:
    def __init__(self, model_id, token):
        self.model = InferenceClientModel(model_id=model_id, token=token)
        self.agent = CodeAgent(tools=[], model=self.model, additional_authorized_imports=["pandas","h2o.automl"])
    
    def train(self, data_path: str, out_directory: str):
        self.agent.run(f"Data file located at {data_path}" \
        "\n\n   Get the train data and test data when the ratio of train data versus test data is 4:1" \
        "\n\n   Try to use the h2o package to solve problem:" \
        "\n\n       1. Import file using h2o.import_file" \
        "\n\n       2. Split data using H2OFrame.split_frame and passing a random seed of H2OFrame.split_frame" \
        "\n\n       3. Initialize h2o.automl.H2OAutoML, arguments of this function shown below:" \
        "\n\n           - max_models: 1" \
        "\n\n           - seed: 1" \
        "\n\n       4. Train and validate supervised models using data acquired by previous step via h2o.automl.H2OAutoML.train, arguments of this function shown below:" \
        "\n\n           - y: \"class\", do not capital the first character"  \
        "\n\n       5. Display the AutoML Leaderboard" \
        "\n\n       6. Save the leader model using download_mojo function, arguments of this function shown below:" \
        f"\n\n           - path: \"{out_directory}\", neither capital the first character nor add subdirectory"  \
        "\n\n           - get_genmodel_jar: True")