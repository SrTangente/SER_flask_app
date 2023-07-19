from flask import Flask, request, jsonify, render_template
from data import prepare_data, get_features_from_single_file
from model import build_model, train_model
from tensorflow.keras.models import load_model
import pickle
import sklearn

# paste your existing code for data processing, feature extraction, and model building here

app = Flask(__name__)

# endpoint for training the model
@app.route('/train', methods=['GET'])
def train():
    global model  # this will make sure we update the global model variable
    print("Loading and processing data...")
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data()
    model = build_model(x_train)  # build the model
    print("Training model...")
    train_model(model, x_train, y_train, x_val, y_val)
    model.save('model.h5')  # save the trained model
    print("Model saved")
    return jsonify({'message': 'Model trained!!'}), 202

# endpoint for making prediction
@app.route('/predict', methods=['GET'])
def predict():
    path = request.args.get('path')
    if not path:
        return jsonify({'error': 'Missing path parameter'}), 400

    x, y = get_features_from_single_file(path)
    if x is None:
        return jsonify({'error': f'No features found for file {path}. Did you train the model?'}), 400

    model_path = 'model.h5'
    model = load_model(model_path)
    if model is None:
        return jsonify({'error': f'No trained model found. Did you train the model?'}), 400

    encoder_file = open("encoder", "rb")
    encoder = pickle.load(encoder_file)
    encoder_file.close()

    y_pred = model.predict([x])
    prediction = encoder.inverse_transform(y_pred)

    return render_template('prediction.html',
                           prediction=prediction.tolist()[0][0],
                           label=y.tolist()[0],
                           audio_path=path)


if __name__ == '__main__':
    app.run(debug=True)
