# preprocessing.py
# Preprocessing module for sentiment analysis

import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources - added explicit error handling
try:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("If the error persists, manually download the resources with:")
    print("import nltk")
    print("nltk.download('punkt')")
    print("nltk.download('stopwords')")
    print("nltk.download('wordnet')")


def load_data(file_path):
    """
    Load the TripAdvisor hotel reviews dataset

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_text(text):
    """
    Clean text by removing special characters, numbers, etc.

    Args:
        text (str): Text to clean

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags if any
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):
    """
    Remove stopwords from text

    Args:
        text (str): Text to process

    Returns:
        str: Text without stopwords
    """
    if not isinstance(text, str) or text == "":
        return ""

    try:
        stop_words = set(stopwords.words('english'))

        # Use simple split instead of word_tokenize to avoid potential issues
        word_tokens = text.split()

        filtered_text = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_text)
    except Exception as e:
        print(f"Error removing stopwords: {e}")
        return text  # Return original text if there's an error


def lemmatize_text(text):
    """
    Lemmatize text

    Args:
        text (str): Text to lemmatize

    Returns:
        str: Lemmatized text
    """
    if not isinstance(text, str) or text == "":
        return ""

    try:
        lemmatizer = WordNetLemmatizer()
        # Use simple split instead of word_tokenize
        word_tokens = text.split()
        lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    except Exception as e:
        print(f"Error lemmatizing text: {e}")
        return text  # Return original text if there's an error


def preprocess_data(df, text_column='Review', label_column='Rating'):
    """
    Preprocess text data for sentiment analysis

    Args:
        df (pandas.DataFrame): DataFrame containing hotel reviews
        text_column (str): Name of the column containing review text
        label_column (str): Name of the column containing ratings

    Returns:
        pandas.DataFrame: DataFrame with preprocessed text
    """
    print("Preprocessing data...")
    print(f"Dataset size: {len(df)} reviews")

    # Display rating distribution
    rating_counts = df[label_column].value_counts().sort_index()
    print("\nRating distribution:")
    for rating, count in rating_counts.items():
        print(f"Rating {rating}: {count} reviews ({count / len(df) * 100:.1f}%)")

    # Check if required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Column {text_column} not found in the dataset")
    if label_column not in df.columns:
        raise ValueError(f"Column {label_column} not found in the dataset")

    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # Clean text
    print("\nCleaning text...")
    df_processed['cleaned_text'] = df_processed[text_column].apply(clean_text)

    # Remove stopwords
    print("Removing stopwords...")
    df_processed['text_no_stopwords'] = df_processed['cleaned_text'].apply(remove_stopwords)

    # Lemmatize text
    print("Lemmatizing text...")
    df_processed['text_lemmatized'] = df_processed['text_no_stopwords'].apply(lemmatize_text)

    # Convert ratings to sentiment labels (for TripAdvisor 1-5 rating scale)
    # Ratings 1-2: Negative (0), Rating 3: Neutral (1), Ratings 4-5: Positive (2)
    print("Converting ratings to sentiment labels...")

    def convert_rating_to_sentiment(rating):
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return 1  # Neutral
        else:
            return 2  # Positive

    df_processed['sentiment'] = df_processed[label_column].apply(convert_rating_to_sentiment)

    # Print sentiment distribution
    sentiment_counts = df_processed['sentiment'].value_counts().sort_index()
    print("\nSentiment distribution:")
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        print(f"{sentiment_labels[i]}: {count} reviews ({count / len(df_processed) * 100:.1f}%)")

    print("\nPreprocessing completed.")
    return df_processed


def tokenize_and_pad(texts, max_words=10000, max_sequence_length=100):
    """
    Tokenize and pad text sequences

    Args:
        texts (list): List of preprocessed texts
        max_words (int): Maximum number of words to keep in the vocabulary
        max_sequence_length (int): Maximum length of sequences

    Returns:
        tuple: (tokenizer, padded_sequences)
    """
    print("Tokenizing and padding text sequences...")

    # Create tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)

    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    print(f"Tokenization completed. Vocabulary size: {len(tokenizer.word_index)}")
    return tokenizer, padded_sequences


def prepare_data_for_training(df, text_column='text_lemmatized', label_column='sentiment',
                              test_size=0.4, max_words=10000, max_sequence_length=100,
                              save_path=None):
    """
    Prepare data for model training

    Args:
        df (pandas.DataFrame): DataFrame with preprocessed text
        text_column (str): Name of the column containing preprocessed text
        label_column (str): Name of the column containing sentiment labels
        test_size (float): Test set size (proportion)
        max_words (int): Maximum number of words to keep in the vocabulary
        max_sequence_length (int): Maximum length of sequences
        save_path (str): Directory to save preprocessed data

    Returns:
        tuple: (X_train, X_test, y_train, y_test, tokenizer)
    """
    print("Preparing data for training...")

    # Tokenize and pad sequences
    tokenizer, padded_sequences = tokenize_and_pad(
        df[text_column].values, max_words, max_sequence_length
    )

    # Get labels
    labels = df[label_column].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=test_size, random_state=42, stratify=labels
    )

    print(f"Data split completed. Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    # Save preprocessed data if path is provided
    if save_path:
        print(f"Saving preprocessed data to {save_path}")
        os.makedirs(save_path, exist_ok=True)

        np.save(os.path.join(save_path, 'X_train.npy'), X_train)
        np.save(os.path.join(save_path, 'X_test.npy'), X_test)
        np.save(os.path.join(save_path, 'y_train.npy'), y_train)
        np.save(os.path.join(save_path, 'y_test.npy'), y_test)

        # Save tokenizer
        import pickle
        with open(os.path.join(save_path, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X_train, X_test, y_train, y_test, tokenizer


def main():
    """Main function to run the preprocessing pipeline"""
    # Define paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_dir, 'data', 'tripadvisor_hotel_reviews.csv')
    processed_data_path = os.path.join(current_dir, 'data', 'processed')

    # Load data
    df = load_data(data_path)

    if df is not None:
        # Preprocess data
        df_processed = preprocess_data(df)

        # Prepare data for training
        prepare_data_for_training(
            df_processed,
            save_path=processed_data_path
        )

        print("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()