import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.pipeline import Pipeline
from smolagents import CodeAgent, InferenceClientModel

class DataAnalysisAgent:
    def __init__(self, model_id, token, class_column, postive_class, negative_class):
        self.model = InferenceClientModel(model_id=model_id, token=token)
        self.agent = CodeAgent(tools=[], model=self.model, additional_authorized_imports=["pandas"])
        self.class_column = class_column
        self.postive_class = postive_class
        self.negative_class = negative_class
        self.numerical_transformer = None
        self.categorical_transformer = None


    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Handle missing values by dropping rows with any missing values
        data = data.dropna()

        classes = data.pop(self.class_column).map({self.postive_class: 1, self.negative_class: 0}).astype('category')

        numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

        if self.numerical_transformer is None:
            self.numerical_transformer = Pipeline(
                steps=[
                    ('discretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')),
                    ('numerical_selector', SelectPercentile(score_func=f_classif, percentile=80))
                ]
            )

        categorical_features = data.select_dtypes(include=['object', 'category']).columns

        if self.categorical_transformer is None:
            self.categorical_transformer = Pipeline(
                steps=[
                    ('encoder', OrdinalEncoder()),
                    ('categorical_selector', SelectPercentile(score_func=chi2, percentile=80))
                ]
            )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, numerical_features),
                ('cat', self.categorical_transformer, categorical_features)
            ])
        
        features_processed = preprocessor.fit_transform(data, classes)
        
        features_df = pd.DataFrame(features_processed, columns=preprocessor.get_feature_names_out())
        
        features_df[self.class_column] = classes

        return features_df

    def extract_features(self, train_data: pd.DataFrame, validate_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_features = self.process_data(train_data)
        validate_features = self.process_data(validate_data)

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