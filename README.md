# Sentiment Analysis of Hotel Reviews Using Deep Learning

## Overview
This project is an implementation of sentiment analysis on hotel reviews using deep learning techniques. The main goal is to classify TripAdvisor hotel reviews into three sentiment categories: Negative, Neutral, and Positive. The project is structured as coursework for COM 671 and leverages LSTM and Bidirectional LSTM neural networks for text classification.

## Features
- Preprocessing of raw hotel review data (cleaning, stopword removal, lemmatization)
- Tokenization and sequence padding for deep learning models
- Training and comparison of LSTM and Bidirectional LSTM models
- Evaluation of model performance (accuracy, precision, recall, F1-score, confusion matrix)
- Prediction of sentiment for new reviews or CSV files
- Visualizations of results and model comparisons

## Project Structure
```
artificial-intelligence-coursework/
├── data/
│   ├── processed/                # Preprocessed data and tokenizer
│   └── tripadvisor_hotel_reviews.csv  # Raw dataset
├── reports/
│   ├── screenshots/              # Model evaluation and comparison plots
│   └── Sentiment Analysis of Hotel Reviews Using Deep Learning.docx
├── src/
│   ├── __init__.py               # Package info
│   ├── evaluate.py               # Model evaluation
│   ├── predict.py                # Sentiment prediction
│   ├── preprocessing.py          # Data preprocessing
│   └── train_model.py            # Model training and comparison
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd artificial-intelligence-coursework
   ```
2. **Install dependencies:**
   This project requires Python 3.7+ and the following Python packages:
   - numpy
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn
   - tensorflow (2.x)
   - nltk

   Install them using pip:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow nltk
   ```
   
   Additionally, download the required NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **Prepare the data:**
   - Place the `tripadvisor_hotel_reviews.csv` file in the `data/` directory (already present).
   - Run the preprocessing pipeline to generate processed data:
     ```bash
     python src/preprocessing.py
     ```

4. **Train the models:**
   ```bash
   python src/train_model.py
   ```
   This will train both LSTM and Bidirectional LSTM models, compare their performance, and save the best model.

5. **Evaluate the model:**
   ```bash
   python src/evaluate.py
   ```
   This will generate evaluation metrics and plots in `reports/screenshots/`.

6. **Make predictions:**
   - For a single review or a CSV file:
     ```bash
     python src/predict.py
     ```
   - Edit `src/predict.py` to change the input review or CSV file as needed.

## File Descriptions
- `src/preprocessing.py`: Cleans and preprocesses the raw review data, tokenizes text, and prepares data for model training.
- `src/train_model.py`: Trains and compares LSTM and Bidirectional LSTM models for sentiment classification.
- `src/evaluate.py`: Evaluates the trained model(s) and generates performance metrics and plots.
- `src/predict.py`: Loads the trained model and tokenizer to predict sentiment for new reviews or datasets.
- `data/processed/`: Contains preprocessed data arrays and the tokenizer.
- `reports/screenshots/`: Contains generated plots (confusion matrix, classification report, model comparison).

## Dataset
- The project uses the [TripAdvisor Hotel Reviews dataset](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) (CSV format).
- Sentiment labels are derived from ratings: 1-2 = Negative, 3 = Neutral, 4-5 = Positive.

## Results
- Model performance metrics and visualizations are saved in `reports/screenshots/` after training and evaluation.
- The best model is saved as `models/best_model.h5` (created after running training script).

## License
This project is for educational purposes as part of COM 671 coursework.

---

*For any issues or questions, please contact the project maintainer.* 