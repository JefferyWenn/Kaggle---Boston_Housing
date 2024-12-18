"""
File: boston_housing_competition.py
Name:
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


def main():
    df = pd.read_csv("train.csv")
    df.head()
    df.info()

    # Plotting all the columns to look at their distributions
    for i in df.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=i, kde=True)
        # plt.show()

    df['medv_log'] = np.log(df['medv'])
    sns.histplot(data=df, x='medv_log', kde=True)

    plt.figure(figsize=(12, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap=cmap)
    # plt.show()

    # Scatterplot to visualize the relationship between AGE and DIS
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='age', y='dis', data=df)
    # plt.show()

    # Scatterplot to visulaize the relationship between RAD and TAX
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='rad', y='tax', data=df)
    # plt.show()

    # Remove the data corresponding to high tax rate
    df1 = df[df['tax'] < 600]
    # Import the required function
    from scipy.stats import pearsonr
    # Calculate the correlation
    print('The correlation between TAX and RAD is', pearsonr(df1['tax'], df1['rad'])[0])

    # Scatterplot to visualize the relationship between INDUS and TAX
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='indus', y='tax', data=df)
    # plt.show()

    # Scatterplot to visulaize the relationship between RM and MEDV
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='rm', y='medv', data=df)
    # plt.show()

    # Scatterplot to visulaize the relationship between LSTAT and MEDV
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='lstat', y='medv', data=df)
    # plt.show()

    # Scatterplot to visualize the relationship between INDUS and NOX
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='indus', y='nox', data=df)
    # plt.show()

    # Scatterplot to visualize the relationship between AGE and NOX
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='age', y='nox', data=df)
    # plt.show()

    # Scatterplot to visualize the relationship between DIS and NOX
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='dis', y='nox', data=df)
    # plt.show()

    # Separate the dependent variable and indepedent variables
    Y = df['medv_log']
    X = df.drop(columns={'medv', 'medv_log'})
    # Add the intercept term
    X = sm.add_constant(X)

    # splitting the data in 70:30 ratio of train to test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # Function to check VIF
    def checking_vif(train):
        vif = pd.DataFrame()
        vif["feature"] = train.columns
        # Calculating VIF for each feature
        vif["VIF"] = [
            variance_inflation_factor(train.values, i) for i in range(len(train.columns))
        ]
        return vif

    print(checking_vif(X_train))

    # Create the model after dropping TAX
    X_train = X_train.drop(columns='tax')
    # Check for VIF
    print(checking_vif(X_train))
    # Create the model using ordinary least squared
    model1 = sm.OLS(y_train, X_train).fit()
    # Get the model summary
    model1.summary()

    # Create the model after dropping columns 'MEDV', 'MEDV_log', 'TAX', 'ZN', 'AGE', 'INDUS' from df DataFrame
    Y = df['medv_log']
    X = df.drop(['zn', 'age', 'indus'], axis=1)
    X = sm.add_constant(X)
    # Splitting the data in 70:30 ratio of train to test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
    # Create the model
    model2 = sm.OLS(y_train, X_train).fit()
    # Get the model summary
    model2.summary()

    residuals = model2.resid
    np.mean(residuals)

    from statsmodels.stats.diagnostic import het_white
    from statsmodels.compat import lzip
    import statsmodels.stats.api as sms
    name = ["F statistic", "p-value"]
    test = sms.het_goldfeldquandt(y_train, X_train)
    lzip(name, test)

    # Predicted values
    fitted = model2.fittedvalues
    # sns.set_style("whitegrid")
    sns.residplot(x=fitted, y=residuals, color="lightblue", lowess=True)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residual")
    plt.title("Residual PLOT")
    # plt.show()

    # Plot histogram of residuals
    sns.histplot(residuals, kde=True)

    # Plot q-q plot of residuals
    import pylab
    import scipy.stats as stats
    stats.probplot(residuals, dist="norm", plot=pylab)
    # plt.show()

    # RMSE
    def rmse(predictions, targets):
        return np.sqrt(((targets - predictions) ** 2).mean())

    # MAPE
    def mape(predictions, targets):
        return np.mean(np.abs((targets - predictions)) / targets) * 100

    # MAE
    def mae(predictions, targets):
        return np.mean(np.abs((targets - predictions)))

    ## R2
    from sklearn.metrics import r2_score

    # Model Performance on test and train data
    def model_pref(olsmodel, x_train, x_test):
        # In-sample Prediction
        y_pred_train = olsmodel.predict(x_train)
        y_observed_train = y_train

        # Prediction on test data
        y_pred_test = olsmodel.predict(x_test)
        y_observed_test = y_test

        print(
            pd.DataFrame(
                {
                    "Data": ["Train", "Test"],
                    "RMSE": [
                        rmse(y_pred_train, y_observed_train),
                        rmse(y_pred_test, y_observed_test),
                    ],
                    "MAE": [
                        mae(y_pred_train, y_observed_train),
                        mae(y_pred_test, y_observed_test),
                    ],
                    "MAPE": [
                        mape(y_pred_train, y_observed_train),
                        mape(y_pred_test, y_observed_test),
                    ],
                    "r2": [
                        r2_score(y_pred_train, y_observed_train),
                        r2_score(y_pred_test, y_observed_test),
                    ],
                }
            )
        )

    # Checking model performance
    model_pref(model2, X_train, X_test)
    # Import the required function
    from sklearn.model_selection import cross_val_score
    # Build the regression model and cross-validate
    linearregression = LinearRegression()
    cv_Score11 = cross_val_score(linearregression, X_train, y_train, cv=10)
    cv_Score12 = cross_val_score(linearregression, X_train, y_train, cv=10,
                                 scoring='neg_mean_squared_error')
    print("RSquared: %0.3f (+/- %0.3f)" % (cv_Score11.mean(), cv_Score11.std() * 2))
    print("Mean Squared Error: %0.3f (+/- %0.3f)" % (-1 * cv_Score12.mean(), cv_Score12.std() * 2))

    coef = model2.params
    pd.DataFrame({'Feature': coef.index, 'Coefs': coef.values})





    # Ensure 'X_test' and 'y_test' are defined and 'coef' contains the model's coefficients
    features = X_test.columns  # This should include the 'const' if using statsmodels' add_constant

    # Initialize predictions with the intercept term
    # Ensure 'const' is the first element in coef if using statsmodels
    predictions = coef[0]  # The intercept

    # Add the contributions of each feature
    for i in range(1, len(coef)):
        predictions += coef[i] * X_test[features[i]]

    print(predictions)


if __name__ == "__main__":
    main()