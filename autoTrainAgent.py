from smolagents import CodeAgent, InferenceClientModel

def generate_model(data_path: str):
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token="")
    
    agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["h2o","h2o.automl"])

    agent.run(f"Data file located at {data_path}" \
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
        "\n\n           - path: \".\\generatedFiles\", do not capital the first character"  \
        "\n\n           - get_genmodel_jar: True")