
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import sklearn 
import joblib 
import boto3 
import pathlib
from io import StringIO
import argparse 
import joblib 
import os
import numpy as np 
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the
    parser.add_argument("--n_estimators", type=int, default=10) 
    parser.add_argument("--random_state", type=int, default=0)

    # Data model and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train_data", type=str, default="train_V-1.csv")
    parser.add_argument("--test_data", type=str, default="test_V-1.csv")

    args, _ = parser.parse_known_args()

    train_df = pd.read_csv(os.path.join(args.train, args.train_data))
    test_df = pd.read_csv(os.path.join(args.test, args.test_data))

    print("[INFO] train data shape: {}".format(train_df.shape))
    print("[INFO] test data shape: {}".format(test_df.shape))

    features = list(train_df.columns)
    label = features.pop(-1)

    X_train = train_df[features].values
    y_train = train_df[label].values

    X_test = test_df[features].values
    y_test = test_df[label].values

    print("[INFO] training model")
    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=1)
    clf.fit(X_train, y_train)

    print("[INFO] saving model")
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

    y_pred_test = clf.predict(X_test)
    print("[INFO] classification report")
    print(classification_report(y_test, y_pred_test))

    print("[INFO] confusion matrix")
    print(confusion_matrix(y_test, y_pred_test))

    print("[INFO] accuracy score")
    print(accuracy_score(y_test, y_pred_test))
