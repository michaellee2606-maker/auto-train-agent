import os
import h2o
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from smolagents import CodeAgent, InferenceClientModel


class MachineLearningAgent:
    def __init__(self, model_id, token, font_path):
        self.fontProps = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = self.fontProps.get_name()
        self.model = InferenceClientModel(model_id=model_id, token=token)
        self.agent = CodeAgent(tools=[], model=self.model, additional_authorized_imports=["pandas","h2o.automl"])
    
    def train(self, train_feature_path: str, out_directory: str):
        self.agent.run(f"Get the train data file which is located at {train_feature_path}" \
        "\n\n   Try to use the h2o package to solve problem:" \
        "\n\n       1. Import file using h2o.import_file" \
        "\n\n       2. Initialize h2o.automl.H2OAutoML, arguments of this function shown below:" \
        "\n\n           - max_models: 1" \
        "\n\n           - seed: 1" \
        "\n\n       3. Train and validate supervised models using data acquired by previous step via h2o.automl.H2OAutoML.train, arguments of this function shown below:" \
        "\n\n           - y: \"class\", do not capital the first character"  \
        "\n\n       4. Display the AutoML Leaderboard" \
        "\n\n       5. Save the leader model using download_mojo function, arguments of this function shown below:" \
        f"\n\n           - path: \"{out_directory}\", neither capital the first character nor add subdirectory"  \
        "\n\n           - get_genmodel_jar: True")

    def generate_report(self, validate_feature_path: str, out_directory: str):
        validate_features = h2o.import_file(validate_feature_path)
        models = [file for file in os.listdir(out_directory) if file.endswith(".zip")]

        for model in models:
            model_path = out_directory + os.sep + model
            imported_model = h2o.import_mojo(model_path)

            validate_result = imported_model.predict(validate_features)

            combined_result = validate_result['predict'].cbind(validate_features['class'])
            grouped_result = combined_result.group_by(['predict', 'class']).count().get_frame()
            pivot_result = grouped_result.pivot(index='class', column='predict', value='nrow')
            confusion_matrix = pivot_result.as_data_frame(use_pandas=True, header=True, use_multi_thread=True)
            
            fig, ax = plt.subplots()
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=confusion_matrix.values, colLabels=confusion_matrix.columns, 
                loc='center', colLoc='center', cellLoc='center')

            # Add a title for the table
            plt.title('混淆矩阵', fontsize=14, fontweight='bold', fontproperties=self.fontProps)

            # Calculate Precision and Recall
            precision = confusion_matrix.iloc[0, 1] / (confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[1, 1])
            recall = confusion_matrix.iloc[0, 2] / (confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[0, 2])

            # Add Precision and Recall to the plot
            plt.figtext(0.5, 0.01, f'坏客户识别准确率: {precision:.2f}, 坏客户误判率: {recall:.2f}', wrap=True, horizontalalignment='center', fontsize=12, fontproperties=self.fontProps)

            # Save the table as a PDF
            report_path = out_directory + os.sep + f'Report - {model}.pdf'
            plt.savefig(report_path)
