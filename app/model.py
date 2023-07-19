import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization


def build_model(x_train):
    model = Sequential()

    # First set of Convolutional and Pooling layers
    model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding="same", activation="relu", input_shape=(x_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

    # Second set of Convolutional and Pooling layers
    model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

    # Third set of Convolutional and Pooling layers
    model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

    # Flattening the output to connect to Dense layers
    model.add(Flatten())

    # First Dense layer
    model.add(Dense(units=32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Output Dense layer
    model.add(Dense(units=7, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

def train_model(model, x_train, y_train, x_val, y_val):
    # Set up early stopping and learning rate reduction

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.025,
        patience=30,
        verbose=1,
        restore_best_weights=True
    )

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=4,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=1e-6)

    callbacks=[early_stopping,learning_rate_reduction]

    model.fit(x_train, y_train, batch_size=64, epochs=80, validation_data=(x_val, y_val), callbacks=callbacks)

    return model