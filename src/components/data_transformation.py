import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")


class DataTransformation:
  def __init__(self):
    self.data_transformation_config=DataTransformationConfig()

  def get_data_transformer_object(self):
    '''
    This function is responsible for data transformation
    '''
    try:
      numerical_columns=["writing_score","reading_score"]
      categorical_columns=[
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course"]
      
      num_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="median")),
          ("scaler",StandardScaler())
        ]
      )

      cat_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="most_frequent")),
          ("one_hot_encoder",OneHotEncoder()),
          ("scaler",StandardScaler(with_mean=False))
        ]
      )

      logging.info(f"Numerical features: {numerical_columns}")
      logging.info(f"Categorical features: {categorical_columns}")
   

      preprocessor=ColumnTransformer(
        [
          ("numerical_pipeline",num_pipeline,numerical_columns),
          ("categorical_pipeline",cat_pipeline,categorical_columns)  
        ]
      )

      return preprocessor
    
    except Exception as e:
      raise CustomException(e,sys)
    
  def initiate_data_tranformation(self,train_path,test_path):
    try:
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)

      logging.info("The training and testing data formation is completed")

      preprocessing_obj=self.get_data_transformer_object()

      target_column_name="math_score"

      input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
      target_feature_train_df=train_df[target_column_name]

      input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
      target_feature_test_df=test_df[target_column_name]

      logging.info("Apply preprocessing on train and test dataframe")

      input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

      train_arr=np.c_[
        input_feature_train_arr, np.array(target_feature_train_df)
      ]
      test_arr=np.c_[
        input_feature_test_arr, np.array(target_feature_test_df)
      ]

      logging.info("Saved preprocessing object")

      save_object(
        file_path=self.data_transformation_config.preprocessor_obj_file_path,
        obj=preprocessing_obj
      )

      return (
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_file_path
      )
    except Exception as e:
      raise CustomException(e,sys)
