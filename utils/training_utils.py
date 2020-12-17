from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def identity(X):
    return X


def build_features_pipeline(config):
    transformers = []
    if config["features"]["one_hot_encode_categorical"]:
        ohe = OneHotEncoder(
            categories=list(config["features"]["categorical_labels"].values())
        )
    else:
        ohe = FunctionTransformer(identity, validate=False)
    transformers.append(
        ("one_hot_encoder", ohe, config["features"]["categorical_features"])
    )
    if config["features"]["normalize_numerical"]:
        scaler = StandardScaler()
    else:
        scaler = FunctionTransformer(identity, validate=False)
    transformers.append(
        ("normalizer", scaler, config["features"]["numerical_features"])
    )
    return ColumnTransformer(transformers)


def build_label_encoder(config):
    users = []
    for user in config["users"]:
        users.append(user)
    label_encoder = LabelEncoder()
    return label_encoder.fit(users)


def build_model(config):
    model = Sequential()
    for layer_name in config["model"]["layers"]:
        layer = config["model"]["layers"][layer_name]
        model.add(
            Dense(
                units=layer["units"],
                activation=layer["activation"],
                name=layer_name,
            )
        )
    model.add(Dense(units=1, activation="sigmoid", name="output"))
    adam = Adam(lr=config["model"]["learning_rate"])
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
