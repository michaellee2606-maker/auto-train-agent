from pydantic import BaseModel, Field
from typing import List

class XGBoostHyperparameters(BaseModel):
    backend: str = Field(default='auto', description="Backend to use for XGBoost, e.g., 'cpu' or 'gpu'.")
    dmatrix_type: str = Field(default='dense', description="Type of DMatrix to use, e.g., 'sparse' or 'dense'.")
    tree_method: str = Field(default='hist', description="Tree construction algorithm, e.g., 'hist', 'exact', or 'gpu_hist'.")
    booster: List[str] = Field(default=['gbtree', 'dart'], description="Type of booster to use, e.g., 'gbtree' or 'dart'.")
    ntrees: List[int] = Field(default=[60, 120, 150], description="Number of trees to grow.")
    max_depth: List[int] = Field(default=[6, 11, 15], description="Maximum depth of a tree.")
    min_rows: List[int] = Field(default=[1, 5, 10], description="Minimum number of rows needed to make a split.")
    reg_alpha: List[float] = Field(default=[0.001, 0.01, 0.1], description="L1 regularization term on weights.")
    reg_lambda: List[float] = Field(default=[0.001, 0.01, 0.1], description="L2 regularization term on weights.")
    learn_rate: List[float] = Field(default=[0.01, 0.1, 0.2], description="Learning rate (eta) for boosting.")
    subsample: List[float] = Field(..., description="Subsample ratio of the training instances.")
    colsample_bytree: List[float] = Field(..., description="Subsample ratio of columns when constructing each tree.")

class XGBoostSearchCriteria(BaseModel):
    strategy: str = Field(default='RandomDiscrete', description="Search strategy, e.g., 'RandomDiscrete' or 'Cartesian'.")
    max_models: int = Field(default=20, description="Maximum number of models to build.")
    stopping_metric: str = Field(default='AUCPR', description="Metric to use for early stopping.")
    stopping_tolerance: float = Field(default=0.05, description="Relative tolerance for metric-based stopping criterion.")
    stopping_rounds: int = Field(default=5, description="Number of rounds to tolerate metric-based stopping criterion.")

class XGBoostResponse(BaseModel):
    hyperparameters: XGBoostHyperparameters
    search_criteria: XGBoostSearchCriteria