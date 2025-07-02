# train_model.py
# Model training module for sentiment analysis

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical


def load_preprocessed_data(data_path):
    """
    Load preprocessed data from disk

    Args:
        data_path (str): Directory containing preprocessed data

    Returns:
        tuple: (X_train, X_test, y_train, y_test, tokenizer)
    """
    print("Loading preprocessed data...")

    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    # Load tokenizer
    with open(os.path.join(data_path, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("Data loaded successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test, tokenizer


def prepare_data_for_model(X_train, X_test, y_train, y_test, num_classes=3):
    """
    Prepare data for model training

    Args:
        X_train, X_test (numpy.ndarray): Training and test features
        y_train, y_test (numpy.ndarray): Training and test labels
        num_classes (int): Number of sentiment classes

    Returns:
        tuple: (X_train, X_test, y_train_cat, y_test_cat)
    """
    # Check if we need to convert labels to categorical
    if len(y_train.shape) == 1 and num_classes > 1:
        print("Converting labels to categorical format...")
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        print(f"y_train_cat shape: {y_train_cat.shape}, y_test_cat shape: {y_test_cat.shape}")
        return X_train, X_test, y_train_cat, y_test_cat

    return X_train, X_test, y_train, y_test


def create_lstm_model(max_features, embedding_dim=128, max_len=100, num_classes=3):
    """
    Create a LSTM model for sentiment classification

    Args:
        max_features (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding space
        max_len (int): Maximum length of input sequences
        num_classes (int): Number of output classes

    Returns:
        tensorflow.keras.models.Sequential: Compiled LSTM model
    """
    print("Creating LSTM model...")

    model = Sequential()

    # Add embedding layer
    model.add(Embedding(max_features, embedding_dim, input_length=max_len))
    model.add(SpatialDropout1D(0.2))

    # Add LSTM layer
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

    # Add dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    if num_classes > 1:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model summary
    model.summary()

    return model


def create_bidirectional_lstm_model(max_features, embedding_dim=128, max_len=100, num_classes=3):
    """
    Create a bidirectional LSTM model for sentiment classification

    Args:
        max_features (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding space
        max_len (int): Maximum length of input sequences
        num_classes (int): Number of output classes

    Returns:
        tensorflow.keras.models.Sequential: Compiled bidirectional LSTM model
    """
    print("Creating Bidirectional LSTM model...")

    model = Sequential()

    # Add embedding layer
    model.add(Embedding(max_features, embedding_dim, input_length=max_len))
    model.add(SpatialDropout1D(0.2))

    # Add bidirectional LSTM layer
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))

    # Add dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    if num_classes > 1:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model summary
    model.summary()

    return model


def train_and_compare_models(X_train, X_test, y_train, y_test, tokenizer, num_classes=3,
                             embedding_dim=128, batch_size=128, epochs=15,
                             models_dir='../models', logs_dir='../logs',
                             screenshots_dir='../reports/screenshots'):
    """
    Train and compare LSTM and bidirectional LSTM models

    Args:
        X_train, X_test (numpy.ndarray): Training and test features
        y_train, y_test (numpy.ndarray): Training and test labels
        tokenizer: Tokenizer used for text processing
        num_classes (int): Number of sentiment classes
        embedding_dim (int): Dimension of the embedding space
        batch_size (int): Training batch size
        epochs (int): Maximum number of epochs to train
        models_dir (str): Directory to save trained models
        logs_dir (str): Directory to save TensorBoard logs
        screenshots_dir (str): Directory to save evaluation plots

    Returns:
        dict: Comparison metrics and best model information
    """
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    # Get sequence length from input shape
    max_len = X_train.shape[1]

    # Get vocabulary size
    max_features = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {max_features}")

    # Create models
    lstm_model = create_lstm_model(max_features, embedding_dim, max_len, num_classes)
    bilstm_model = create_bidirectional_lstm_model(max_features, embedding_dim, max_len, num_classes)

    # Define callbacks
    timestamp = int(time.time())

    lstm_checkpoint = ModelCheckpoint(
        filepath=os.path.join(models_dir, 'lstm_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    bilstm_checkpoint = ModelCheckpoint(
        filepath=os.path.join(models_dir, 'bilstm_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    lstm_tensorboard = TensorBoard(
        log_dir=os.path.join(logs_dir, f'lstm_{timestamp}')
    )

    bilstm_tensorboard = TensorBoard(
        log_dir=os.path.join(logs_dir, f'bilstm_{timestamp}')
    )

    # Train LSTM model
    print("Training LSTM model...")
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[lstm_checkpoint, early_stopping, lstm_tensorboard]
    )

    # Train bidirectional LSTM model
    print("Training Bidirectional LSTM model...")
    bilstm_history = bilstm_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[bilstm_checkpoint, early_stopping, bilstm_tensorboard]
    )

    # Load best models
    lstm_model = load_model(os.path.join(models_dir, 'lstm_model.h5'))
    bilstm_model = load_model(os.path.join(models_dir, 'bilstm_model.h5'))

    # Evaluate models
    lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test, verbose=1)
    bilstm_loss, bilstm_accuracy = bilstm_model.evaluate(X_test, y_test, verbose=1)

    print(f"LSTM Model - Test Loss: {lstm_loss:.4f}, Test Accuracy: {lstm_accuracy:.4f}")
    print(f"BiLSTM Model - Test Loss: {bilstm_loss:.4f}, Test Accuracy: {bilstm_accuracy:.4f}")

    # Determine best model
    if lstm_accuracy >= bilstm_accuracy:
        best_model = 'lstm'
        best_model_path = os.path.join(models_dir, 'lstm_model.h5')
        print("LSTM model performed better.")
    else:
        best_model = 'bilstm'
        best_model_path = os.path.join(models_dir, 'bilstm_model.h5')
        print("Bidirectional LSTM model performed better.")

    # Save best model
    best_model_copy_path = os.path.join(models_dir, 'best_model.h5')
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, best_model_copy_path)
        print(f"Best model saved as {best_model_copy_path}")

    # Plot training and validation accuracy for both models
    plt.figure(figsize=(12, 5))

    # Plot training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history.history['accuracy'], label='LSTM Training')
    plt.plot(lstm_history.history['val_accuracy'], label='LSTM Validation')
    plt.plot(bilstm_history.history['accuracy'], label='BiLSTM Training')
    plt.plot(bilstm_history.history['val_accuracy'], label='BiLSTM Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history.history['loss'], label='LSTM Training')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Validation')
    plt.plot(bilstm_history.history['loss'], label='BiLSTM Training')
    plt.plot(bilstm_history.history['val_loss'], label='BiLSTM Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(screenshots_dir, 'model_comparison.png'))
    print(f"Model comparison plot saved to {os.path.join(screenshots_dir, 'model_comparison.png')}")

    # Save model comparison results
    comparison_results = {
        'lstm': {
            'accuracy': float(lstm_accuracy),
            'loss': float(lstm_loss)
        },
        'bilstm': {
            'accuracy': float(bilstm_accuracy),
            'loss': float(bilstm_loss)
        },
        'best_model': best_model
    }

    return comparison_results


def main():
    """Main function to train and compare models"""
    # Define paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_dir, 'data', 'processed')
    models_dir = os.path.join(current_dir, 'models')
    logs_dir = os.path.join(current_dir, 'logs')
    screenshots_dir = os.path.join(current_dir, 'reports', 'screenshots')

    # Load preprocessed data
    X_train, X_test, y_train, y_test, tokenizer = load_preprocessed_data(data_path)

    # Print sentiment distribution in the training and test sets
    if len(np.unique(y_train)) <= 3:
        print("\nSentiment distribution in training set:")
        for i, sentiment in enumerate(['Negative', 'Neutral', 'Positive']):
            count = np.sum(y_train == i)
            print(f"{sentiment}: {count} ({count / len(y_train) * 100:.1f}%)")

        print("\nSentiment distribution in test set:")
        for i, sentiment in enumerate(['Negative', 'Neutral', 'Positive']):
            count = np.sum(y_test == i)
            print(f"{sentiment}: {count} ({count / len(y_test) * 100:.1f}%)")

    # Prepare data for model training (convert labels to categorical for multi-class classification)
    X_train, X_test, y_train, y_test = prepare_data_for_model(X_train, X_test, y_train, y_test)

    # Train and compare models
    comparison_results = train_and_compare_models(
        X_train, X_test, y_train, y_test, tokenizer,
        models_dir=models_dir,
        logs_dir=logs_dir,
        screenshots_dir=screenshots_dir
    )

    print("Model training and comparison completed!")
    print(f"Best model: {comparison_results['best_model'].upper()}")
    print(f"LSTM accuracy: {comparison_results['lstm']['accuracy']:.4f}")
    print(f"BiLSTM accuracy: {comparison_results['bilstm']['accuracy']:.4f}")


if __name__ == "__main__":
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    main()