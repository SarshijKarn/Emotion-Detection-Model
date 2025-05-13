# Emotion Recognition

**Emotion Recognition** is a cutting-edge deep learning project designed to detect and classify human emotions based on facial expressions. Using a Convolutional Neural Network (CNN), the model is trained on the FER2013 dataset and can accurately recognize seven distinct emotions:  
**Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**.

## Installation

 **Create a Virtual Environment**
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
    (dont use )                                  Remove-Item -Recurse -Force .\venv

 
### Train the Model
download it from here:  https://www.kaggle.com/datasets/msambare/fer2013

Train the CNN model on the FER2013 dataset:

python model_maker.py

After training, the model will be saved as **`emotion_model.h5`**.

### Run Real-Time Emotion Detection
Start the webcam-based emotion detection system:


python main.py

Press **`q`** to exit the webcam feed.
