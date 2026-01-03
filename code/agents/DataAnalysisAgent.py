import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.pipeline import Pipeline
from smolagents import CodeAgent, InferenceClientModel

class DataAnalysisAgent:
    def __init__(self, model_id, token, class_column, positive_class, negative_class):
        self.model = InferenceClientModel(model_id=model_id, token=token)
        self.agent = CodeAgent(tools=[], model=self.model, additional_authorized_imports=["pandas"])
        self.class_column = class_column
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.preprocessor = None

    def get_numerical_transformers(self) -> dict:
        bin_edges = self.preprocessor.named_transformers_['num'].named_steps['discretizer'].bin_edges_
        features = self.preprocessor.named_transformers_['num'].named_steps['discretizer'].feature_names_in_

        return dict(zip(features, bin_edges))

    def get_categorical_transformers(self) -> dict:
        encodings = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].categories_
        features = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].feature_names_in_

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
            numerical_transformer = Pipeline(
                steps=[
                    ('discretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')),
                    ('numerical_selector', SelectPercentile(score_func=f_classif, percentile=80))
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ('encoder', OrdinalEncoder()),
                    ('categorical_selector', SelectPercentile(score_func=chi2, percentile=80))
                ]
            )
        
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            features_processed = self.preprocessor.fit_transform(data, classes)

            return features_processed, self.preprocessor.get_feature_names_out()
        else:
            numerical_transformers, categorical_transformers = self.get_transfomers()

            merged_columns = np.hstack((numerical_features.values, categorical_features.values))

            transformed_columns = []

            # Apply numerical transformations based on training transformers
            for numerical_feature in numerical_features:
                transformed_columns.append('num__' + numerical_feature)
                data[numerical_feature] = np.digitize(data[numerical_feature], numerical_transformers[numerical_feature])

            # Apply categorical transformations based on training transformers
            for categorical_feature in categorical_features:
                transformed_columns.append('cat__' + categorical_feature)
                data[categorical_feature] = data[categorical_feature].map(categorical_transformers[categorical_feature])

            # Rename columns to match transformed feature names
            data.rename(columns=dict(zip(merged_columns, transformed_columns)), inplace=True)

            return data[self.preprocessor.get_feature_names_out()], self.preprocessor.get_feature_names_out()


    def process_data(self, data: pd.DataFrame, train_flag: bool) -> pd.DataFrame:
        # Handle missing values by dropping rows with any missing values
        data = data.dropna()

        classes = data.pop(self.class_column).map({self.positive_class: 1, self.negative_class: 0}).astype('category')

        features_processed, columns = self.transform_data(data, classes, train_flag)
        
        features_df = pd.DataFrame(features_processed, columns=columns)
        
        features_df[self.class_column] = classes

        return features_df

    def extract_features(self, train_data: pd.DataFrame, validate_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_features = self.process_data(train_data, True)
        validate_features = self.process_data(validate_data, False)

        return train_features, validate_features

    # Analyze the training and validation data, extract features, and save to new CSV files.
    def analyze(self, train_data_path: str, validate_data_path: str) -> tuple[str, str]:
        train_features_path = os.path.dirname(train_data_path) + os.sep + "train_features.csv"
        validate_features_path = os.path.dirname(validate_data_path) + os.sep + "validate_features.csv"

        train_data = pd.read_csv(train_data_path)
        validate_data = pd.read_csv(validate_data_path)
        
        train_features, validate_features = self.extract_features(train_data, validate_data)

        train_features.to_csv(train_features_path, index=False)
        validate_features.to_csv(validate_features_path, index=False)

        return train_features_path, validate_features_path