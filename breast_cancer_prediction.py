"""
Converted from: c:/Users/itzan/Downloads/Breast Cancer Prediction (1).ipynb
Description: Breast cancer prediction example using Decision Tree and Logistic Regression.
"""

# importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def main():
    # importing the dataset (try project data.csv, then common Downloads path)
    possible_paths = [
        r'C:\Users\itzan\Downloads\data (1).csv',
        r'C:\Users\itzan\Downloads\data.csv',
        'data.csv',
    ]
    df = None
    for p in possible_paths:
        if os.path.exists(p):
            print(f"Loading dataset from: {p}")
            df = pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError('data.csv not found in project root or Downloads. Provide dataset path.')
    print('\nDataset head:')
    print(df.head())

    # map diagnosis to numeric for analysis and modeling (M=1, B=0)
    if df['diagnosis'].dtype == object:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    # drop rows with missing target
    df.dropna(subset=['diagnosis'], inplace=True)

    # Data Preprocessing Part 1
    # dropping unnecessary columns
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

    # checking for the missing values
    print('\nMissing values:')
    print(df.isnull().sum())

    # checking the data types of the columns
    print('\nDtypes:')
    print(df.dtypes)

    # checking the data description
    print('\nDescribe:')
    print(df.describe())

    # Exploratory Data Analysis
    print('\nCorrelation with diagnosis:')
    print(df.corr()['diagnosis'].sort_values())

    # bar plot for the number of diagnosis
    plt.figure(figsize=(5, 5))
    sns.barplot(x=df['diagnosis'].value_counts().index, y=df['diagnosis'].value_counts().values)
    plt.title('Diagnosis counts')
    plt.show()

    # create a heatmap to check the correlation
    plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=True)
    plt.title('Feature correlation heatmap')
    plt.show()

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(['diagnosis'], axis=1), df['diagnosis'], test_size=0.3, random_state=42
    )

    # ensure there are at least two classes to train classifiers
    if df['diagnosis'].nunique() < 2:
        print('\nDataset contains only one class in the target. Cannot train classifiers requiring multiple classes.')
        return

    # Using Decision Tree Classifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    # predicting the diagnosis
    y_pred = dtree.predict(X_test)

    # Model Evaluation for Decision Tree
    print('\nDecision Tree - sample predictions vs actual:')
    print('Predicted values: ', y_pred[:10])
    print('Actual values:    ', list(y_test[:10]))

    print('\nDecision Tree accuracy:')
    print(dtree.score(X_test, y_test))

    # Using logistic regression
    logmodel = LogisticRegression(max_iter=10000)
    logmodel.fit(X_train, y_train)

    yhat = logmodel.predict(X_test)

    # Model Evaluation for Logistic Regression
    print('\nLogistic Regression - sample predictions vs actual:')
    print('Predicted values: ', yhat[:10])
    print('Actual values:    ', list(y_test[:10]))

    print('\nLogistic Regression accuracy:')
    print(logmodel.score(X_test, y_test))

    # Conclusion
    print('\nConclusion:')
    print('Compare the printed accuracy values to evaluate models.')


if __name__ == '__main__':
    main()
