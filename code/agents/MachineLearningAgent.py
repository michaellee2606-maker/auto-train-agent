import os
import h2o
from h2o.estimators import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch
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

        xgboost_params = {
            'backend': 'auto',
            'dmatrix_type': 'dense',
            'tree_method': 'hist',
            'booster': ['gbtree', 'dart'],
            'ntrees': [60, 400],
            'max_depth': [6, 11, 15],
            'min_rows': [1, 5, 10],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1],
            'learn_rate': [0.01, 0.1]
        }

        search_criteria = {
            'strategy': 'RandomDiscrete',
            'max_models': 10,
            'seed': 42,
            'stopping_metric': 'AUCPR',
            'stopping_tolerance': 0.001,
            'stopping_rounds': 5
        }

        xgboost_grid = H2OGridSearch(
            grid_id='xgboost_grid',
            model=H2OXGBoostEstimator,
            hyper_params=xgboost_params,
            search_criteria=search_criteria,
            parallelism=0
        )

        xgboost_grid.train(
            y=self.class_column,
            training_frame=train_features,
            nfolds=5,
            fold_assignment='Stratified',
        )

        xgboost_model = xgboost_grid.get_grid(sort_by='f2', decreasing=True)[0]

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
            precision = confusion_matrix.iloc[1, 2] / (confusion_matrix.iloc[0, 2] + confusion_matrix.iloc[1, 2])
            recall = confusion_matrix.iloc[1, 2] / (confusion_matrix.iloc[1, 1] + confusion_matrix.iloc[1, 2])

            # Add Precision and Recall to the plot
            plt.figtext(0.5, 0.01, f'召回率: {recall:.2f}, 精确率: {precision:.2f}', wrap=True, horizontalalignment='center', fontsize=12, fontproperties=self.fontProps)

            # Save the table as a PDF
            report_path = out_directory + os.sep + f'Report - {model}.pdf'
            plt.savefig(report_path)
