FROM python:3.7-slim-buster
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN bash setup.sh
CMD ["python", "get_spotify_data.py", "&&", "python", "train_spotipy_model", "&&", "streamlit","run", "tsne_plot.py"]
