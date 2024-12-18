import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Constants - filenames for data set
TRAIN_PATH = 'boston_housing/train.csv'
TEST_PATH = 'boston_housing/test.csv'

def main():
    try:
        # Data cleaning
        train_data = pd.read_csv(TRAIN_PATH)
        test_data = pd.read_csv(TEST_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Extract true labels (target variable)
    y = train_data['medv']
    train_data.pop('ID')
    id = test_data.pop('ID')

    # Extract features
    feature_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
    x_train = train_data[feature_names]
    x_test = test_data[feature_names]

    # Handle missing values and standardize features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), feature_names)
        ]
    )

    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    # Split train data for cross-validation
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning for RandomForestRegressor
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    forest = RandomForestRegressor(random_state=42)
    grid_search_rf = GridSearchCV(estimator=forest, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error')
    grid_search_rf.fit(x_train_split, y_train_split)
    best_forest = grid_search_rf.best_estimator_

    # Hyperparameter tuning for GradientBoostingRegressor
    param_grid_gb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    gbr = GradientBoostingRegressor(random_state=42)
    grid_search_gb = GridSearchCV(estimator=gbr, param_grid=param_grid_gb, cv=5, scoring='neg_mean_squared_error')
    grid_search_gb.fit(x_train_split, y_train_split)
    best_gbr = grid_search_gb.best_estimator_

    # Hyperparameter tuning for KNeighborsRegressor
    param_grid_knn = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
    knn = KNeighborsRegressor()
    grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='neg_mean_squared_error')
    grid_search_knn.fit(x_train_split, y_train_split)
    best_knn = grid_search_knn.best_estimator_

    # Hyperparameter tuning for SVR
    param_grid_svr = {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5],
        'kernel': ['linear', 'rbf']
    }
    svr = SVR()
    grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, scoring='neg_mean_squared_error')
    grid_search_svr.fit(x_train_split, y_train_split)
    best_svr = grid_search_svr.best_estimator_

    # Hyperparameter tuning for XGBRegressor
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb = XGBRegressor(random_state=42)
    grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
    grid_search_xgb.fit(x_train_split, y_train_split)
    best_xgb = grid_search_xgb.best_estimator_

    # Ridge Regression with polynomial features
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly_features.fit_transform(x_train_split)
    x_val_poly = poly_features.transform(x_val_split)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_poly, y_train_split)

    # Stacking ensemble model
    estimators = [
        ('forest', best_forest),
        ('gbr', best_gbr),
        ('knn', best_knn),
        ('svr', best_svr),
        ('xgb', best_xgb)
    ]
    stacking = StackingRegressor(estimators=estimators, final_estimator=ElasticNet())

    stacking.fit(x_train_split, y_train_split)
    stacking_train_predictions = stacking.predict(x_train_split)
    stacking_val_predictions = stacking.predict(x_val_split)

    # Evaluate the stacking model
    print(f"Stacking Training R^2 Score: {stacking.score(x_train_split, y_train_split)}")
    print(f"Stacking Validation R^2 Score: {stacking.score(x_val_split, y_val_split)}")
    print(f"Stacking Validation RMSE: {mean_squared_error(y_val_split, stacking_val_predictions, squared=False)}")

    # Cross-validation score
    cv_score = cross_val_score(stacking, x_train, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation RMSE: {-cv_score.mean() ** 0.5}")

    # Make predictions on test data
    test_predictions = stacking.predict(x_test)
    out_file(test_predictions, id, 'stacking_predictions.csv')

def out_file(predictions, id, filename):
    """
    :param predictions: numpy.array, a list-like data structure that stores 0's and 1's
    :param filename: str, the filename you would like to write the results to
    """
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        for i, ans in enumerate(predictions):
            out.write(f"{id[i]},{ans:.8f}\n")
    print('===============================================')

if __name__ == '__main__':
    main()
