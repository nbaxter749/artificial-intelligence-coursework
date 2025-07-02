# evaluate.py
# Evaluation module for sentiment analysis

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


def load_test_data(data_path):
    """
    Load test data from disk

    Args:
        data_path (str): Directory containing preprocessed data

    Returns:
        tuple: (X_test, y_test)
    """
    print("Loading test data...")

    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    print(f"Test data loaded. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test


def load_best_model(models_dir):
    """
    Load the best trained model

    Args:
        models_dir (str): Directory containing saved models

    Returns:
        tensorflow.keras.models.Model: The best trained model
    """
    print("Loading best model...")

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
    return model


def prepare_test_data(X_test, y_test, num_classes=3):
    """
    Prepare test data for evaluation

    Args:
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        num_classes (int): Number of sentiment classes

    Returns:
        tuple: (X_test, y_test, y_test_categorical)
    """
    # Check if we need to convert labels to categorical for multi-class classification
    if len(y_test.shape) == 1 and num_classes > 1:
        y_test_categorical = to_categorical(y_test, num_classes=num_classes)
        print(f"Test labels converted to categorical. Shape: {y_test_categorical.shape}")
        return X_test, y_test, y_test_categorical

    return X_test, y_test, y_test


def evaluate_model(model, X_test, y_test, y_test_cat, class_names, screenshots_dir):
    """
    Evaluate model performance

    Args:
        model: Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Original test labels
        y_test_cat (numpy.ndarray): Categorical test labels (if applicable)
        class_names (list): Names of the sentiment classes
        screenshots_dir (str): Directory to save evaluation plots

    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating model on TripAdvisor hotel reviews...")
    print(f"Test set size: {len(X_test)} reviews")

    # Make predictions
    y_pred_prob = model.predict(X_test)

    # Convert probabilities to class labels
    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
        # Multi-class
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        # Binary
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    # For multi-class classification
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(screenshots_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")

    # Plot classification report
    plt.figure(figsize=(12, 8))
    report_df = pd.DataFrame(report).transpose()

    # Remove accuracy, macro avg, and weighted avg for the heatmap
    report_df = report_df.iloc[:-3]

    # Select only precision, recall, and f1-score columns
    report_df = report_df[['precision', 'recall', 'f1-score']]

    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Classification Report')
    plt.tight_layout()
    classification_report_path = os.path.join(screenshots_dir, 'classification_report.png')
    plt.savefig(classification_report_path)
    print(f"Classification report heatmap saved to {classification_report_path}")

    # Collect evaluation metrics
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    return evaluation_metrics


def main():
    """Main function to evaluate the model"""
    # Define paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_dir, 'data', 'processed')
    models_dir = os.path.join(current_dir, 'models')
    screenshots_dir = os.path.join(current_dir, 'reports', 'screenshots')

    # Ensure screenshots directory exists
    os.makedirs(screenshots_dir, exist_ok=True)

    # Define class names
    class_names = ['Negative', 'Neutral', 'Positive']

    # Load test data
    X_test, y_test = load_test_data(data_path)

    # Prepare test data for evaluation
    X_test, y_test, y_test_cat = prepare_test_data(X_test, y_test, len(class_names))

    # Load best model
    model = load_best_model(models_dir)

    # Evaluate model
    evaluation_metrics = evaluate_model(
        model, X_test, y_test, y_test_cat, class_names, screenshots_dir
    )

    print("Model evaluation completed!")
    print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()