from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from utils.io_utils import yaml_loader, get_data, save_model, save_transformers
from utils.training_utils import (
    build_features_pipeline,
    build_label_encoder,
    build_model,
    early_stopping,
)

config = yaml_loader("./config/config.yml")
data = get_data(config)

features_pipeline = build_features_pipeline(config)
label_encoder = build_label_encoder(config)
model = build_model(config)

X = data.loc[:, data.columns != config["features"]["target"]]
y = data[config["features"]["target"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=12345, test_size=0.25, shuffle=True, stratify=y
)

X_train = features_pipeline.fit_transform(X_train)
X_test = features_pipeline.transform(X_test)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

model.fit(
    x=X_train,
    y=y_train,
    validation_data=[X_test, y_test],
    epochs=config["model"]["epochs"],
    batch_size=config["model"]["batch_size"],
    callbacks=[early_stopping],
)

y_pred_proba = model.predict(X_test)
y_pred = model.predict_classes(X_test)

print(f"F1 score: {f1_score(y_test,y_pred)}")
print(f"ROC AUC: {roc_auc_score(y_test,y_pred_proba)}")
print(f"Confusion matrix: \n{confusion_matrix(y_test, y_pred)}")
print(f"Classification report: \n{classification_report(y_test, y_pred)}")

save_transformers(features_pipeline, label_encoder, config)
save_model(model, config)
