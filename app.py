import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import base64
import cv2
import io
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import webbrowser

# Load the trained model
MODEL_PATH = "model_alex_net.h5"  # Make sure your model is in the same directory
# MODEL_PATH = "VGG16_Model.h5"  # Make sure your model is in the same directory
model = load_model(MODEL_PATH, compile=False)

# Define class labels
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Define image size
IMAGE_SIZE = (48, 48)

# Set up Spotify API credentials
CLIENT_ID = "your id"
CLIENT_SECRET = "ur secret"
REDIRECT_URI = "http://127.0.0.1:8888/callback"  # Ensure this is registered in Spotify Developer Dashboard

# Authenticate for public playlists (Client Credentials)
sp_public = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# Authenticate for user-specific playlists (OAuth)
sp_user = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                                    client_secret=CLIENT_SECRET,
                                                    redirect_uri=REDIRECT_URI,
                                                    scope="playlist-read-private"))

# Create Flask app
app = Flask(__name__)


# Emotion-to-genre mapping (with Telugu songs)
emotion_to_query = {
    "happy": "Telugu upbeat",
    "sad": "Telugu slow",
    "neutral": "My favourites",
    "angry": "Mass Telugu Beats",
    "fear": "Telugu devotional songs"
}

def get_playlist_for_emotion(emotion, fav_playlist_name="My favourites"):
    """Fetches a Spotify playlist based on the detected emotion."""
    
    if emotion.lower() == "surprise":
        return get_favorite_playlist(fav_playlist_name)  # Open user's favorite playlist
    
    query = emotion_to_query.get(emotion.lower(), "Telugu chill")
    results = sp_public.search(q=f"{query} playlist", type="playlist", limit=1)

    if results["playlists"]["items"]:
        playlist = results["playlists"]["items"][0]
        return {"name": playlist["name"], "url": playlist["external_urls"]["spotify"]}
    else:
        return {"error": "No playlist found"}

  

def get_favorite_playlist(fav_playlist_name):
    """Fetches the user's favorite playlist by name."""
    playlists = sp_user.current_user_playlists()
    n=playlists["items"][0]
    if n:
      return {"name": n["name"], "url": n["external_urls"]["spotify"]}
    
    return {"error": "Favorite playlist not found"}

# Helper function to process image data from base64
def decode_image(img_data):
    img_bytes = base64.b64decode(img_data.split(",")[1])  # Remove header
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMAGE_SIZE)  # Resize to model input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    print(img.shape)
    # img = np.expand_dims(img, axis=-1)  # Add channel dimension
    # img = img / 255.0  # Normalize
    return img

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

# Route for handling image classification
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON data from front-end
    img_data = data.get("image")  # Extract base64 image

    if not img_data:
        return jsonify({"error": "No image received"}), 400

    # Decode and preprocess image
    img = decode_image(img_data)

    # Predict emotion
    result = model.predict(img)
    y_pred = np.argmax(result[0])
    predicted_emotion = classes[y_pred]
    print("Predicted Emotion: ",predicted_emotion)
    fav_playlist_name = "My Favorites"  # Replace with your actual playlist name

    # Fetch playlist based on emotion (or open favorite if "surprise")
    playlist_info = get_playlist_for_emotion(predicted_emotion, fav_playlist_name)
    print("playlist details: ",playlist_info)
    playlist_url=playlist_info["url"]
    webbrowser.open(playlist_url)
    return jsonify({"emotion": predicted_emotion})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
