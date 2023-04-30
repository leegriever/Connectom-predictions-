import numpy as np
import pandas as pd
import correlation_permuation_test
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso

def run_prediction_pipeline(X, y, model, params_grid = None, splitter = None):
    """
    ----------
    Parameters
    ----------
    X - pd.DataFrame
        Features for prediction model
        n X m array, where n is the number of samples (subjects) and m is the number of features
    y - np.ndarray
        Scores to predict for all subejcts
        n X 1 array
    model - sklearn object
        Prediction model to use 

    ----------
    Optional
    ----------
    params_grid - dict
        Parameters grid for grid search cross validation. If None, using the model's default values.
    splitter - sklearn.model_selection.Kfold/GroupKfold object. Deafult is None.
         sklearn splitter for cross-validation. If None, using 10-fold cross-validation
    
    ----------
    Returns
    ----------
    mean_r - float
        mean correlation between true and predicted target values across cross-validation folds
    mean_pval - float
        mean p-value of the correlation, calculated using a permutation test, across cross-validation folds
    predicted_scores - np.ndarray
        Predicted scores for all subjects
        n X 1 array

    """

    if splitter == None:
        splitter = KFold(n_splits = 10, shuffle=True, random_state=0)
    
    
    predicted_scores = np.zeros(y.shape)
    r_vec = [] 
    pval_vec = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        print("\n*** FOLD #{fold} ***".format(fold=fold+1))
        # Get relevent data slices for train-test split
        X_train = X.iloc[train_indices,:]
        X_test = X.iloc[test_indices, :]
        y_train = y[train_indices]
        y_test = y[test_indices]

        # Normalization of features and behavioral scores 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = scaler.fit_transform(y_train).ravel()
        y_test = scaler.transform(y_test).ravel()

        # Feature selection
        max_features = int(X_train.shape[1]* 0.05) # top 5% features
        selector = SelectFromModel(estimator=RandomForestRegressor(criterion = "squared_error", n_estimators = 100, max_features = 'sqrt', random_state=0, n_jobs=-1), max_features = max_features) 
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        if params_grid == None:
            best_estimator = model.fit(X_train,y_train)

        else: 
            clf = GridSearchCV(model, params_grid, scoring = "neg_mean_squared_error", cv = 10, n_jobs=-1)
            clf.fit(X_train,y_train)
            best_estimator = clf.best_estimator_ # best esitmator chosen by grid search
            print('Config: %s' % clf.best_params_)

        y_train_predicted = best_estimator.predict(X_train)
        y_predicted = best_estimator.predict(X_test)
        
        print('Train MSE: %.3f' % mean_squared_error(y_train,  y_train_predicted))
        print('Test MSE: %.3f' % mean_squared_error(y_test, y_predicted))

        predicted_scores[test_indices] = scaler.inverse_transform(y_predicted.reshape(-1,1)) 
        r, pval, _ = correlation_permuation_test(y_predicted, y_test)
        r_vec.append(r)
        pval_vec.append(pval)
    
    mean_r = np.mean(r_vec)
    mean_pval = np.mean(pval_vec)
    
    return mean_r, mean_pval, predicted_scores

# our code 
Glasser_edges_indices = np.load("data/Glasser_conn_mat_158_subj.npy")

X = []
indices = np.triu_indices(360, k = 1)

for i in range(Glasser_edges_indices.shape[2]):
    X.append(Glasser_edges_indices[:,:,i][indices])

y = pd.read_csv("data/general_scores_158_subj.csv", header = None).to_numpy()
X = np.array(X)
X =     pd.DataFrame(X)
mean_r, mean_pval, predicted_scores = run_prediction_pipeline(X, y, model = Lasso)
print("mearn r: ", mean_r, ", mean pval: ", mean_pval, " predicted_scores: ", predicted_scores)