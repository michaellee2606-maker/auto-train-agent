import os
import h2o
from h2o.estimators import H2OXGBoostEstimator
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from smolagents import CodeAgent, InferenceClientModel


class MachineLearningAgent:
    def __init__(self, model_id, token, font_path, class_column):
        self.fontProps = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = self.fontProps.get_name()
        self.model = InferenceClientModel(model_id=model_id, token=token)
        self.agent = CodeAgent(tools=[], model=self.model, additional_authorized_imports=["pandas","h2o.automl"])
        self.class_column = class_column
    
    def train(self, train_feature_path: str, out_directory: str):
        train_features = h2o.import_file(train_feature_path)
        train_features[self.class_column] = train_features[self.class_column].asfactor()

        xgboost_model = H2OXGBoostEstimator(
            booster='gbtree',
            backend='cpu',
            dmatrix_type='dense',
            tree_method='hist',
            nfolds=5,
            ntrees=60,
            max_depth=6,
            min_rows=3,
            min_split_improvement=1e-3,
            reg_alpha=1e-2,
            reg_lambda=1.0,
            learn_rate=0.05,
            stopping_metric='AUCPR',
            stopping_rounds=5,
            stopping_tolerance=1e-3,
            score_tree_interval=4,
            seed=0
        )

        xgboost_model.train(y=self.class_column, training_frame=train_features)

        xgboost_model.download_mojo(path=out_directory, get_genmodel_jar=True)

    def generate_report(self, validate_feature_path: str, out_directory: str):
        validate_features = h2o.import_file(validate_feature_path)
        actual = validate_features.pop(self.class_column).asfactor()

        models = [file for file in os.listdir(out_directory) if file.endswith(".zip")]

        for model in models:
            model_path = out_directory + os.sep + model
            imported_model = h2o.import_mojo(model_path)

            validate_result = imported_model.predict(validate_features)

            combined_result = validate_result['predict'].cbind(actual)
            grouped_result = combined_result.group_by(['predict', self.class_column]).count().get_frame()
            pivot_result = grouped_result.pivot(index=self.class_column, column='predict', value='nrow')
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
            recall = confusion_matrix.iloc[0, 1] / (confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[0, 2])

            # Add Precision and Recall to the plot
            plt.figtext(0.5, 0.01, f'召回率: {recall:.2f}, 精确率: {precision:.2f}', wrap=True, horizontalalignment='center', fontsize=12, fontproperties=self.fontProps)

            # Save the table as a PDF
            report_path = out_directory + os.sep + f'Report - {model}.pdf'
            plt.savefig(report_path)
