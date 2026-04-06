from pydantic import BaseModel, Field
from typing import List

class FeatureSelectionResponse(BaseModel):
    """The response of feature selection"""
    content: str = Field(description="The content field contains a detailed explanation of the feature selection process")
    chosen_features: List[int] = Field(description="The chosen_features field is a list of indices representing the selected features")