import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from xgboost.sklearn import XGBRegressor  

### Function that creates Out-of-fold predictions for an algorithm clf

def get_predictions(clf, X_train, y_train, X_test):

    print("Training of algorithm started.")
    print()

    kf = KFold(n_splits = 5)

    oof_predictions = np.array([])
    oof_predictions_test = []

    i = 0

    for train_index, test_index in kf.split(X_train):

        X_tr, X_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]

        start = time.time()

        clf.fit(X_tr, y_tr)

        end = time.time()

        print("Training round {} of 5".format(i + 1))
        print("Training time: {} seconds".format(end - start))
        print("Best score: {}".format(clf.best_score_))
        print("Optimal params: {}".format(clf.best_estimator_))
        print()

        i += 1

        oof_predictions = np.append(oof_predictions, clf.predict(X_te))
        oof_predictions_test.append(clf.predict(X_test))

    return oof_predictions, np.asarray(oof_predictions_test).mean(axis = 0)

### Importing cleaned data

id_data = pd.read_csv("data/test.csv")
id_data = id_data["Id"]

train_data = pd.read_csv("data/train_clean.csv")
ntrain = len(train_data)
test_data = pd.read_csv("data/test_clean.csv")

full_data = pd.concat([train_data, test_data], sort = False, ignore_index = True)

### Applying the log to SalePrice and 1stFlrSF

full_data["1stFlrSF"] = full_data["1stFlrSF"].apply(lambda x: np.log(x))

train_data = full_data.iloc[:ntrain, :]
test_data = full_data.iloc[ntrain:, :]

X_train = train_data.drop("SalePrice", axis = 1).values
y_train = train_data["SalePrice"].values
X_test = test_data.drop("SalePrice", axis = 1).values

log_y_train = np.log(y_train)

print()
print("--- Linear regression with GridSearch ---")
print()

lm_params = {'fit_intercept': [True, False], 'normalize': [True, False]}

lm = GridSearchCV(LinearRegression(), lm_params, n_jobs = -1, cv = 5).fit(X_train, log_y_train)

# lm_pred = lm.predict(X_test)

X_train_lm, X_test_lm = get_predictions(lm, X_train, log_y_train, X_test)

print()
print("--- Lasso regression with GridSearch ---")
print()

lasso_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03]}

llm = GridSearchCV(Lasso(max_iter = 500), lasso_params, n_jobs = -1, cv = 5).fit(X_train, log_y_train)

# llm_pred = llm.predict(X_test)

X_train_llm, X_test_llm = get_predictions(llm, X_train, log_y_train, X_test)

print()
print("--- Ridge regression with GridSearch ---")
print()

ridge_params = {'alpha':[200, 230, 250,265, 270, 275, 290, 300, 500]}

rlm = GridSearchCV(Ridge(), ridge_params, n_jobs = -1, cv = 5).fit(X_train, log_y_train)

# rlm_pred = rlm.predict(X_test)

X_train_rlm, X_test_rlm = get_predictions(rlm, X_train, log_y_train, X_test)

### Simple mean of algorithm predictions

# mean_log_predictions = np.concatenate((lm_pred.reshape(-1,1), llm_pred.reshape(-1,1), rlm_pred.reshape(-1,1)), 
# 	axis = 1).mean(axis = 1)

# predictions = np.exp(mean_log_predictions)

# output = pd.DataFrame({"Id": id_data.values, "SalePrice": predictions})
# output.to_csv("data/ensemble_predictions.csv", index = False)

### XGBoost from the three regression methods

X_train_xgb = np.concatenate((X_train_lm.reshape(-1,1), X_train_llm.reshape(-1,1), X_train_rlm.reshape(-1,1)), axis = 1)
X_test_xgb = np.concatenate((X_test_lm.reshape(-1,1), X_test_llm.reshape(-1,1), X_test_rlm.reshape(-1,1)), axis = 1)

xgb_params = {'objective':['reg:linear'],
              'learning_rate': [0.03, 0.05, 0.07],
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]
              }

gbm = GridSearchCV(XGBRegressor(), xgb_params, n_jobs = -1, cv = 5)

print("--- Training of XGBoost started ---")
print()

start = time.time()

gbm.fit(X_train_xgb, log_y_train)

end = time.time()

print("Training time: {} seconds".format(end - start))
print("Best score: {}".format(gbm.best_score_))
print("Optimal params: {}".format(gbm.best_estimator_))
print()

log_predictions = gbm.predict(X_test_xgb)

predictions = np.exp(log_predictions)

output = pd.DataFrame({"Id": id_data.values, "SalePrice": predictions})
output.to_csv("data/xgb_predictions.csv", index = False)


