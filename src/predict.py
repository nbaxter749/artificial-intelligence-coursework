# predict.py
# Prediction module for sentiment analysis

import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from src.preprocessing import clean_text, remove_stopwords, lemmatize_text, load_data


def load_model_and_tokenizer(models_dir, data_path):
    """
    Load the trained model and tokenizer

    Args:
        models_dir (str): Directory containing the saved model
        data_path (str): Directory containing the tokenizer

    Returns:
        tuple: (model, tokenizer)
    """
    print("Loading model and tokenizer...")

    # Load model
    model_path = os.path.join(models_dir, 'best_model.h5')
    if not os.path.exists(model_path):
        # If best_model.h5 doesn't exist, try lstm_model.h5 or bilstm_model.h5
        if os.path.exists(os.path.join(models_dir, 'lstm_model.h5')):
            model_path = os.path.join(models_dir, 'lstm_model.h5')
        elif os.path.exists(os.path.join(models_dir, 'bilstm_model.h5')):
            model_path = os.path.join(models_dir, 'bilstm_model.h5')
        else:
            raise FileNotFoundError("No trained model found in the models directory")

    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Load tokenizer
    tokenizer_path = os.path.join(data_path, 'tokenizer.pickle')
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded")

    return model, tokenizer


def preprocess_text(text, tokenizer, max_len=100):
    """
    Preprocess text for prediction

    Args:
        text (str): Text to preprocess
        tokenizer: Tokenizer to use for tokenization
        max_len (int): Maximum sequence length

    Returns:
        numpy.ndarray: Preprocessed text as a padded sequence
    """
    # Clean text
    cleaned_text = clean_text(text)

    # Remove stopwords
    text_no_stopwords = remove_stopwords(cleaned_text)

    # Lemmatize text
    lemmatized_text = lemmatize_text(text_no_stopwords)

    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences([lemmatized_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    return padded_sequences


def predict_sentiment(text, model, tokenizer, class_names=None, max_len=100):
    """
    Predict sentiment of a text

    Args:
        text (str): Text to analyze
        model: Trained sentiment analysis model
        tokenizer: Tokenizer used for preprocessing
        class_names (list): Names of sentiment classes
        max_len (int): Maximum sequence length

    Returns:
        tuple: (predicted_class, confidence)
    """
    # Default class names
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']

    # Preprocess text
    preprocessed_text = preprocess_text(text, tokenizer, max_len)

    # Make prediction
    prediction = model.predict(preprocessed_text)

    # Process prediction based on model output shape
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        # Multi-class classification
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        predicted_class = class_names[predicted_class_idx]
    else:
        # Binary classification
        confidence = prediction[0][0]
        predicted_class = class_names[1] if confidence >= 0.5 else class_names[0]

    return predicted_class, float(confidence)


def predict_from_csv(file_path, model, tokenizer, text_column='Review', class_names=None, max_len=100, limit=None):
    """
    Predict sentiment for reviews in a CSV file

    Args:
        file_path (str): Path to the CSV file
        model: Trained sentiment analysis model
        tokenizer: Tokenizer used for preprocessing
        text_column (str): Name of the column containing review text
        class_names (list): Names of sentiment classes
        max_len (int): Maximum sequence length
        limit (int): Maximum number of reviews to process

    Returns:
        pandas.DataFrame: DataFrame with reviews and predicted sentiments
    """
    # Default class names
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']

    # Load data
    df = load_data(file_path)

    # Limit the number of reviews if specified
    if limit and limit < len(df):
        df = df.sample(limit, random_state=42)

    # Check if text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the CSV file")

    print(f"Predicting sentiment for {len(df)} reviews...")

    # Create lists to store results
    predicted_classes = []
    confidences = []

    # Process each review
    for idx, row in df.iterrows():
        review = row[text_column]
        predicted_class, confidence = predict_sentiment(
            review, model, tokenizer, class_names, max_len
        )
        predicted_classes.append(predicted_class)
        confidences.append(confidence)

        # Print progress for large datasets
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} reviews")

    # Add predictions to DataFrame
    df['predicted_sentiment'] = predicted_classes
    df['confidence'] = confidences

    return df


def main():
    """Main function to demonstrate prediction functionality"""
    # Define paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(current_dir, 'models')
    data_path = os.path.join(current_dir, 'data', 'processed')
    csv_path = os.path.join(current_dir, 'data', 'tripadvisor_hotel_reviews.csv')

    # Define class names
    class_names = ['Negative', 'Neutral', 'Positive']

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(models_dir, data_path)

    # Example 1: Predict sentiment for a single review
    sample_review = "The hotel room was spacious and clean, but the staff was not very helpful. " \
                    "The location was great, close to many attractions."

    predicted_class, confidence = predict_sentiment(
        sample_review, model, tokenizer, class_names
    )

    print("\nExample 1: Predict sentiment for a single review")
    print(f"Review: {sample_review}")
    print(f"Predicted Sentiment: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    # Example 2: Predict sentiment for reviews in a CSV file
    if os.path.exists(csv_path):
        print("\nExample 2: Predict sentiment for reviews in a CSV file")

        # Predict sentiment for a limited number of reviews from TripAdvisor dataset
        results_df = predict_from_csv(
            csv_path, model, tokenizer, text_column='Review', limit=10
        )

        # Display results
        print("\nPrediction Results:")
        print(results_df[['Review', 'predicted_sentiment', 'confidence']].head())

        # Calculate sentiment distribution
        sentiment_counts = results_df['predicted_sentiment'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} ({count / len(results_df) * 100:.1f}%)")
    else:
        print(f"CSV file not found: {csv_path}")


if __name__ == "__main__":
    main()