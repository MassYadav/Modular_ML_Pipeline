from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle
from sklearn.metrics import r2_score

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj )
    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for name, model in models.items():
            try:
                logging.info(f"Training model: {name}")
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[name] = test_model_score
                logging.info(f"{name} R2 score: {test_model_score}")
            except Exception as model_err:
                logging.error(f"Error training {name}: {model_err}")
                report[name] = None  # mark as failed instead of breaking everything

        return report   # ✅ Must always return dictionary (even if some models fail)

    except Exception as e:
        raise CustomException(e, sys)
