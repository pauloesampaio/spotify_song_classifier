import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_features(dataframe, features):
    to_normalize = dataframe[features].copy()
    scaler = StandardScaler()
    scaled_featuers = pd.DataFrame(scaler.fit_transform(to_normalize))
    scaled_featuers.columns = to_normalize.columns
    return scaled_featuers


def get_ohe(dataframe, features):
    dummies_df = pd.concat(
        [pd.get_dummies(data=dataframe[w], prefix=w) for w in features], axis=1
    )
    return dummies_df
