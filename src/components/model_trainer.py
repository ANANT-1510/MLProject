import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR
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
        "Decision Tree":DecisionTreeRegressor(),
        "Random Forest":RandomForestRegressor(),
        "Gradient Boosting":GradientBoostingRegressor(),
        "Linear Regression":LinearRegression(),
        "AdaBoost Regressor":AdaBoostRegressor(),
        "K-Neighbors":KNeighborsRegressor(),
        "SVM":SVR()
      }

      params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['sqrt','log2'],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                
                "SVM":{
                  'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                  # 'degree':[1, 3, 5, 10],
                  'gamma':['scale', 'auto']
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    # 'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
                
            }

      model_report:dict=evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models,params=params)
      
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