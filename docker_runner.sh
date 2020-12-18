echo "GETTING DATA"
python get_spotify_data.py
echo "TRAINING MODEL"
python train_spotipy_model.py 
echo "LOADING TSNE"
streamlit run tsne_plot.py