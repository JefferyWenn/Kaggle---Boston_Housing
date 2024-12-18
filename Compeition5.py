import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

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

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    forest = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train_split, y_train_split)

    best_forest = grid_search.best_estimator_

    # Ridge Regression with polynomial features to reduce complexity
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly_features.fit_transform(x_train_split)
    x_val_poly = poly_features.transform(x_val_split)
    ridge = Ridge(alpha=1.0)

    ridge.fit(x_train_poly, y_train_split)
    ridge_train_predictions = ridge.predict(x_train_poly)
    ridge_val_predictions = ridge.predict(x_val_poly)

    # Ensemble model
    ensemble = VotingRegressor(estimators=[
        ('forest', best_forest),
        ('ridge', ridge)
    ])

    ensemble.fit(x_train_split, y_train_split)
    ensemble_train_predictions = ensemble.predict(x_train_split)
    ensemble_val_predictions = ensemble.predict(x_val_split)

    # Evaluate the ensemble model
    print(f"Ensemble Training R^2 Score: {ensemble.score(x_train_split, y_train_split)}")
    print(f"Ensemble Validation R^2 Score: {ensemble.score(x_val_split, y_val_split)}")
    print(f"Ensemble Validation RMSE: {mean_squared_error(y_val_split, ensemble_val_predictions, squared=False)}")

    # Cross-validation score
    cv_score = cross_val_score(ensemble, x_train, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation RMSE: {-cv_score.mean() ** 0.5}")

    # Make predictions on test data
    test_predictions = ensemble.predict(x_test)
    out_file(test_predictions, id, 'ensemble_forest_ridge.csv')

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
