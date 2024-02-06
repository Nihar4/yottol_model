from flask import Flask, request, jsonify
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.models import load_model


app = Flask(__name__)

@app.route('/train', methods=['GET'])
def train_model():
    try:
        # Code for training the model
        num_words = 10  # Consider the top 10,000 most common words in the dataset
        max_len = 2     # Limit the length of each review to 200 words

        # Load the IMDB dataset
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

        # Pad sequences to ensure consistent input size
        x_train = pad_sequences(x_train, maxlen=max_len)
        x_test = pad_sequences(x_test, maxlen=max_len)

        # Define the LSTM model
        model = Sequential()
        model.add(Embedding(num_words, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

        # Save the trained model
        model.save('lstm_model.h5')

        return 'Model trained successfully'
    except Exception as e:
        return f'Error training model: {str(e)}', 500

@app.route('/predict', methods=['GET'])
def predict():
    try:
        num_words = 10  # Consider the top 10,000 most common words in the dataset
        max_len = 2     # Limit the length of each review to 200 words

        # Load the IMDB test dataset
        (_, _), (x_test, _) = imdb.load_data(num_words=num_words)

        # Pad sequences to ensure consistent input size
        x_test = pad_sequences(x_test, maxlen=max_len)

        # Load the trained LSTM model
        model = load_model('lstm_model.h5')

        # Make predictions on the test data
        predictions = model.predict(x_test)

        # Example: Print the first 10 predictions
        result = [{'sentiment': 'Positive' if pred > 0.5 else 'Negative'} for pred in predictions[:10]]

        return jsonify(result)
    except Exception as e:
        return f'Error predicting: {str(e)}', 500



