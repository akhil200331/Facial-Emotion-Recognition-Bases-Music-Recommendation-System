# 🎵 Facial Emotion Recognition-Based Music Recommendation System

This project is a deep learning-based application that recognizes facial emotions and recommends music accordingly. It aims to create an emotionally intelligent system that enhances a user's mood by suggesting songs that match or elevate their current emotional state.

## 🚀 Features

- 🎭 **Emotion Detection**: Real-time facial emotion recognition using CNN and transfer learning (e.g., VGG16).
- 🎧 **Smart Music Recommendation**: Suggests songs from genres and moods mapped to detected emotions.
- 💬 **Emotion-Playlist Mapping**:
  - **Happy** → Upbeat songs  
  - **Sad** → Slower or calming songs  
  - **Neutral** → User’s favorite playlist  
  - **Angry** → Upbeat or rap music  
  - **Fear** → Religious/devotional songs  
  - **Surprise** → New song recommendations  
  - **Disgust** → Soothing instrumentals  

## 📁 Dataset

- **FER2013** (Facial Expression Recognition 2013)
  - 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
  - Train/test split with imbalanced classes
  - Preprocessed for input into a deep learning model

## 🧠 Model

- CNN-based architecture using **Transfer Learning**
- Base Model: `AlexNet with Self Attention` or `ResNet50`
- Loss Function: `categorical_crossentropy`
- Metrics: Accuracy, Precision, Recall, F1-score
- Target Accuracy: ~95% with F1-score close to 1

## 🛠️ Technologies Used

- Python, TensorFlow, Keras
- OpenCV (for real-time face detection)
- Matplotlib / Seaborn (for visualization)
- Spotify API (or local playlists for recommendation)
- Flask (for backend)
- HTML/CSS/JavaScript (for frontend)

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/akhil200331/Facial-Emotion-Recognition-Bases-Music-Recommendation-System.git
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

4. Open your browser and go to `http://localhost:5000`.

## 📊 Evaluation

- Accuracy: ~90%
- F1-Score: ~0.95
- Class imbalance handled using:
  - Data augmentation

## 📌 Future Improvements

- Integrate personalized user preferences and history
- Real-time emotion detection from video streams
- Multilingual or regional music recommendations (e.g., Telugu, Hindi, etc.)
- Add voice-controlled interaction
