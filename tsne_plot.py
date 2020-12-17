from utils.io_utils import get_data, load_transformers, yaml_loader, load_model
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import pandas as pd
import altair as alt
import streamlit as st

config = yaml_loader("./config/config.yml")
features_transformer, _ = load_transformers(config)
model = load_model(config)
extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
data = get_data(config)
labels = data[config["features"]["target"]]
X = features_transformer.transform(data)

feature_vectors = extractor.predict(X)
tsne_xy = TSNE(random_state=12345).fit_transform(feature_vectors)
tsne_dataframe = pd.DataFrame(tsne_xy, columns=["x", "y"])
tsne_dataframe["labels"] = labels
tsne_dataframe["song"] = data["name"]

tsne_plot = (
    alt.Chart(tsne_dataframe)
    .mark_circle(size=75)
    .encode(x="x", y="y", color="labels", tooltip=["x", "y", "song", "labels"])
    .interactive()
    .properties(title="t-SNE plot (hover for details)", width=750, height=500)
)
st.altair_chart(tsne_plot)