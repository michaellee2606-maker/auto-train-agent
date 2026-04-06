import os
import logging
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
# from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
from utils.FeatureSelector import FeatureSelector
from sklearn.feature_selection import f_classif, chi2
from sklearn.pipeline import Pipeline
# from sklearn.ensemble import GradientBoostingClassifier
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

class DataAnalysisAgent:
    def __init__(self, feature_selection_analysis_model_id, feature_selection_format_model_id, token, 
                 class_column, positive_class, negative_class):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.feature_selection_analysis_model_id = feature_selection_analysis_model_id
        self.feature_selection_format_model_id = feature_selection_format_model_id
        self.token = token

        self.class_column = class_column
        self.positive_class = positive_class
        self.negative_class = negative_class

        self.reports_dict = {}

        self.numerical_selector = FeatureSelector(
            feature_selection_analysis_model_id = feature_selection_analysis_model_id, 
            feature_selection_format_model_id = feature_selection_format_model_id, 
            token = token, feature_type='numerical', score_func=f_classif, reports_dict=self.reports_dict)
        self.categorical_selector = FeatureSelector(
            feature_selection_analysis_model_id = feature_selection_analysis_model_id, 
            feature_selection_format_model_id = feature_selection_format_model_id, 
            token = token, feature_type='categorical', score_func=chi2, reports_dict=self.reports_dict)
        
        self.numerical_transformer = None
        self.categorical_transformer = None

        self.name_of_features_selected = None

    def get_numerical_transformers(self) -> dict:
        bin_edges = self.numerical_transformer.named_steps['discretizer'].bin_edges_
        features = self.numerical_transformer.named_steps['discretizer'].feature_names_in_

        return dict(zip(features, bin_edges))

    def get_categorical_transformers(self) -> dict:
        encodings = self.categorical_transformer.named_steps['encoder'].categories_
        features = self.categorical_transformer.named_steps['encoder'].feature_names_in_

        encodings_list = []

        for encoding in encodings:
            encodings_list.append(dict(zip((encoding), range(len(encoding)))))
            
        return dict(zip(features, encodings_list))

    def get_transfomers(self) -> tuple[dict, dict]:
        numerical_transformers = self.get_numerical_transformers()
        categorical_transformers = self.get_categorical_transformers()

        return numerical_transformers, categorical_transformers


    def transform_data(self, data: pd.DataFrame, classes: pd.Series, train_flag: bool) -> tuple[pd.DataFrame, list[str]]:
        numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = data.select_dtypes(include=['object', 'category']).columns

        if train_flag:
            self.numerical_transformer = Pipeline(
                steps=[
                    # ('discretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')),
                    ('numerical_selector', self.numerical_selector)
                ]
            )

            self.categorical_transformer = Pipeline(
                steps=[
                    ('encoder', OrdinalEncoder()),
                    ('categorical_selector', self.categorical_selector)
                ]
            )
        
            numerical_features_processed = self.numerical_transformer.fit_transform(data[numerical_features], classes)
            categorical_features_processed = self.categorical_transformer.fit_transform(data[categorical_features], classes)

            features_processed = np.hstack((numerical_features_processed, categorical_features_processed))
            self.name_of_features_selected = np.hstack((self.numerical_transformer.get_feature_names_out(), self.categorical_transformer.get_feature_names_out()))

            self.logger.info(f"Features which is selected by univariate selection: {self.name_of_features_selected}")

            # classifier = GradientBoostingClassifier()

            # efs = EFS(classifier,
            #         min_features=8,
            #         max_features=15,
            #         scoring='f1',
            #         print_progress=True,
            #         n_jobs=-1,
            #         cv=5)

            # efs = efs.fit(features_processed, classes)

            # self.name_of_features_selected = [self.preprocessor.get_feature_names_out()[i] for i in efs.best_idx_]

            # self.logger.info(f"Features which is selected by ExhaustiveFeatureSelector: {self.name_of_features_selected}")

            return features_processed, self.name_of_features_selected
        else:
            # numerical_transformers, categorical_transformers = self.get_transfomers()
            categorical_transformers = self.get_categorical_transformers()

            # Apply numerical transformations based on training transformers
            # for numerical_feature in numerical_features:
            #     data[numerical_feature] = np.digitize(data[numerical_feature], numerical_transformers[numerical_feature])

            # Apply categorical transformations based on training transformers
            for categorical_feature in categorical_features:
                data[categorical_feature] = data[categorical_feature].map(categorical_transformers[categorical_feature])

            return data[self.name_of_features_selected], self.name_of_features_selected


    def process_data(self, data: pd.DataFrame, train_flag: bool) -> pd.DataFrame:
        # Handle missing values by dropping rows with any missing values
        data = data.dropna()

        classes = data.pop(self.class_column).map({self.positive_class: 1, self.negative_class: 0}).astype('category')

        # Handle class imbalance using RandomOverSampler when training
        if train_flag:
            imbalanced_sampler = RandomOverSampler(random_state=42)
            data, classes = imbalanced_sampler.fit_resample(data, classes)

        features_processed, columns = self.transform_data(data, classes, train_flag)
        
        features_df = pd.DataFrame(features_processed, columns=columns)
        
        features_df[self.class_column] = classes

        return features_df

    def extract_features(self, train_data: pd.DataFrame, validate_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_features = self.process_data(train_data, True)
        validate_features = self.process_data(validate_data, False)

        return train_features, validate_features

    # Analyze the training and validation data, extract features, and save to new CSV files.
    def analyze(self, train_data_path: str, validate_data_path: str):
        train_features_path = os.path.dirname(train_data_path) + os.sep + "train_features.csv"
        validate_features_path = os.path.dirname(validate_data_path) + os.sep + "validate_features.csv"

        train_data = pd.read_csv(train_data_path)
        validate_data = pd.read_csv(validate_data_path)
        
        train_features, validate_features = self.extract_features(train_data, validate_data)

        train_features.to_csv(train_features_path, index=False)
        validate_features.to_csv(validate_features_path, index=False)

        return train_features_path, validate_features_path, self.reports_dict