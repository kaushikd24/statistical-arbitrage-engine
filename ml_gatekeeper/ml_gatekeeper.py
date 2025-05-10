import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import sklearn
import joblib

fodder_df = pd.read_csv("data/trade_ml_fodder.csv")

#splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X  = fodder_df.drop(columns = ["label"])
Y = fodder_df["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#training the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

from sklearn.metrics import classification_report, roc_auc_score

#predicting the labels and probabilities
Y_pred = model.predict(X_test)
Y_proba = model.predict_proba(X_test)[:,1]

#printing the classification report and the AUC score
print(classification_report(Y_test, Y_pred))
print("AUC Score:", roc_auc_score(Y_test, Y_proba))

#saving the model
joblib.dump(model, "ml_gatekeeper/RFC_model.pkl")

