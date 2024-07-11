import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder


def standard_box_plot(X):
    # Boxplot
    plt.figure(figsize=(15, 10))
    X.boxplot()
    plt.xticks(rotation=90)
    plt.title("Boxplot")
    plt.show()


def standard_histogram(X):
    # Histogram
    plt.figure(figsize=(15, 10))
    X.hist()
    plt.title("Histogram")
    plt.show()


def multiple_plots(X):
    # Boxplot
    standard_box_plot(X)
    # Histogram
    standard_histogram(X)


def standard_OHE(df):
    """
    One-hot encodes the categorical columns in a DataFrame.
    """
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    one_hot_df = pd.DataFrame(
        one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns)
    )

    # Join the one-hot encoded columns to the original DataFrame
    df_encoded = df.join(one_hot_df)

    # Drop the original categorical columns (redundant information)
    df_encoded = df_encoded.drop(categorical_columns, axis=1)
    return df_encoded


def balance_with_SMOTE(x_train):
    """
    Balances the dataset using the Synthetic Minority Over-sampling Technique (SMOTE).
    """
    smote = SMOTE()
    x_train, y_train = smote.fit_resample(x_train, y_train)
    return x_train, y_train
