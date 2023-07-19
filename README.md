# Flask API for Speech Emotion Recognition
This API offers emotion recognition from audio files. It uses Machine Learning and Natural Language Processing to process and classify audio files, and return the predicted emotion.

### Building the container
For building the container, change your directory to this folder and execute the following command

docker build -t ser_app .

#### Run the flask app
docker run -p 5000:5000 ser_app

### Run jupyter lab to visualize the notebook and launch the API
docker run -p 8888:8888 -p 5555:5555 ser_app

### Visualize report
Open jupyterlab by copying one of th urls that will be showed in your terminal. Example:

http://127.0.0.1:8888/lab?token=5282b22ca12ac415df0ce38312463f7c120633a1299ec82b

You can find the report by opening the file Report_SER.ipynb on the left window of the jupyter lab interface

### Flask API Overview
The API consists of two main endpoints: 
1. `/train`: To train the model using audio files
2. `/predict`: To predict the emotion from an audio file

### Train the model

To train the model, you need to use the /train endpoint. It's a GET endpoint and doesn't need any parameters.

Example:

http://localhost:5555/train

After the model training, it will save the model as model.h5.

### Predict the emotion

To predict the emotion from an audio file, use the /predict endpoint with a query parameter, path, which is the path of the audio file.

Example:

http://localhost:5555/predict?path=path_to_audio_file

The endpoint will return an HTML page showing the predicted emotion, the actual emotion label, and an audio player to play the audio file.