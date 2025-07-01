import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()

  def initiate_model_trainer(self,train_arr,test_arr):
    try:
      logging.info("Splitting training and testing input data")

      X_train,y_train,X_test,y_test=(
        train_arr[:,:-1],
        train_arr[:,-1],
        test_arr[:,:-1],
        test_arr[:,-1]
      )

      models={
        "RandomForest":RandomForestRegressor(),
        "LinearRegression":LinearRegression(),
        "DecisionTreeRegression":DecisionTreeRegressor(),
        "K-Neighbors":KNeighborsRegressor(),
        "GradientBoosting":GradientBoostingRegressor(),
        "AdaBoost Classifier":AdaBoostRegressor(),
        "RandomForest Classifier":RandomForestRegressor()
      }

      model_report:dict=evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models)
      
      ## To get the best model score from the dictionary
      best_model_score=max(sorted(model_report.values()))

      ## To get best model name from dictionary
      best_model_name=list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]

      best_model=models[best_model_name]

      if best_model_score < 0.6:
        raise CustomException("No best model found")
      
      logging.info("Best model found")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )

      predicted=best_model.predict(X_test)
      r2_square=r2_score(y_test,predicted)
      
      return r2_square
    except Exception as e:
      raise CustomException(e,sys)