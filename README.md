# Spotify song classifier

This is a project that builds a simple classifier using spotify track features to classify songs into playlists of 2 users. 

I'm using the [Spotipy](https://spotipy.readthedocs.io/en/2.16.1/) python library to access the Spotify's API endpoints and tensorflow to build a simple classifier. I'm also generating a t-SNE plot just for visualize the results.

## Config

All configurations are on the `./config/config.yml` file. The main groups are:

- paths configurations: Paths to where to save data, models and so on
- users: Spotify user IDs you want to use
- model: Instructions on how to build the neural network (number of layers, activation function)
- features: Which features to use and how to perform simple feature pre processing

You will also need a credentials file in order to connect with Spotify's API. I'm providing a template on `./credentials/credentials.json`
## Spotify features

Spotify offers a lot of information. We'll user mainly 3 endpoints:
- [User's saved tracks](https://developer.spotify.com/console/get-current-user-saved-tracks/): These will get the track ID for all songs the current user saved (aka liked)
- [Track features](https://developer.spotify.com/documentation/web-api/reference/tracks/get-several-audio-features/): For each one of the liked tracks, get the basic features (key, danceability, acousticness, etc)
- [Track audio analysis](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-analysis/): For each one of the liked tracks, get audio analysis (pitches, timbre, loudness, etc). This particular endpoint get tracks one by one, so this part is a bit slow. As we are getting less than 500 songs, it is still feasible.

In order to access these endpoints, you need to register on the [Spotify for developers](https://developer.spotify.com/), so you have a CLIENT_ID, CLIENT_SECRET and REDIRECT_URI, which should be stored on the `./credentials/credentials.json`. You also need to have a user tokens for each one of the users that you want to access the songs. The [Spotipy documentation](https://spotipy.readthedocs.io/en/2.16.1/#authorization-code-flow) describes this process quite well and I suggest you to take a look at it and follow a couple of examples on how to authenticate.

## Tensorflow model

I wanted a simple neural network just to use as example with students, so I create a builder that, based on what is set on the config file, builds a simple MLP network. As I'm working with only 2 users, this is a binary classification task. 

I'm also adding an early stopping callback, in order to spot the training if the validation loss doesn't decrease in 5 consecutive epochs.

## Running

There are 3 main commands:

- `python get_spotify_data.py`: This will get the data form spotify for the users on `config.yml`
- `python train_spotipy_model.py`: This performs feature preprocessing/engineering, builds a MLP model, trains it and print a couple of metrics (AUC ROC, F1 score, confusion matrix)
- `streamlit run tsne_plot.py`: Starts a simple webserver with t-SNE interactive visualization, to check class separation, just for fun. Similar to the image below:

![](https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-12-18-spotify_song_classifier/tsne_plot.jpg)
## Docker

I'm providing a `Dockerfile` image and a `docker-compose.yml` in case you want to run in a container. One caveat - you need to have the tokens already cached. This limitation is due to the fact that when you authenticate for the first time, it opens a python `raw_input` for you to input the URL provided by the authenticator, which doesn't work well on docker. So run the `get_data` once locally to get the tokens and then you can use docker. 

## API rate limit

As many APIs, there is a rate limit, so if your `get_data` starts to throw 429 or 503, most likely you hit the API too much. Wait for a bit before running it again. I never had this problem since I'm just getting around 500 songs, but I heard some issues about it.

## Closing thoughts

This was made with educational purposes, mainly to explain simple MLP classification algorithms (There's even a [medium post about it](https://medium.com/@paulo_sampaio/https-medium-com-paulo-sampaio-classification-351a0e3592e9)). Also this is about music taste and there is a huge overlap and diversity in it, so don't expect 100% accuracy - i'm getting about 80%. Evaluate the misclassification an you will have fun - most of the times they make total sense, since even though you are a huge [Iron Maiden](https://en.wikipedia.org/wiki/Iron_Maiden) fan, from time to time you might have actually liked an [ABBA](https://en.wikipedia.org/wiki/ABBA) song!
