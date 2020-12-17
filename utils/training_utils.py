from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def identity(X):
    """Simple identity function

    Args:
        X (any): Any input

    Returns:
        Any: Returns input
    """
    return X


def build_features_pipeline(config):
    """Helper to build feature engineering pipeline

    Args:
        config (dict): Configuration dictionary

    Returns:
        sklearn.ColumnTransformer: Feature engineering pipeline
    """
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
    """Helper to build label encoder

    Args:
        config (dict): Configuration dictionary

    Returns:
        sklearn.LabelEncoder: Label encoder
    """
    users = []
    for user in config["users"]:
        users.append(user)
    label_encoder = LabelEncoder()
    return label_encoder.fit(users)


def build_model(config, n_features):
    inputs = Input(shape=(n_features,), name="input")
    for i, (layer_name) in enumerate(config["model"]["layers"]):
        layer_info = config["model"]["layers"][layer_name]
        current_layer = Dense(
            units=layer_info["units"],
            activation=layer_info["activation"],
            name=layer_name,
        )
        if i == 0:
            x = current_layer(inputs)
        else:
            x = current_layer(x)
    outputs = Dense(units=1, activation="sigmoid", name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)
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
