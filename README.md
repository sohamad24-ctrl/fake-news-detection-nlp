# fake-news-detection-nlp
Fake News Detection using NLP, TF-IDF and Logistic Regression

## Project Overview
This project is a machine learning–based Fake News Detection system that classifies news text as **Real**, **Fake**, or **Uncertain** using Natural Language Processing techniques.

The model uses **TF-IDF vectorization** and **Logistic Regression**, followed by **probability calibration** to improve prediction reliability for short user inputs.

## Project Goal

The goal of this project is to build a machine learning system that detects fake news using Natural Language Processing techniques and probability-calibrated classification.

## Technologies Used
- Python
- scikit-learn
- pandas
- numpy
- joblib

## Project Files
- `train_model.py` – trains the machine learning model
- `train_calibration.py` – calibrates the probability outputs
- `predict_calibrated.py` – predicts whether news is real or fake
- `short_inputs.csv` – small dataset used for calibration

## How to Run
Install dependencies: pip install -r requirements.txt

Train the model: python train_model.py

Run predictions: python predict_calibrated.py


Then enter any news-like sentence in the terminal to see the prediction.

## Dataset
The project uses the **Fake and True News dataset from Kaggle** along with a small manually created dataset for calibration.

## Model Pipeline

1. Text preprocessing (cleaning, normalization)
2. TF-IDF vectorization
3. Logistic Regression classifier
4. Probability calibration (Platt Scaling)
5. Threshold-based classification

## Future Improvements
- Use transformer models such as BERT for improved accuracy
- Add multilingual support
- Build a web-based user interface
- Integrate real-time news verification APIs
  
## Example Usage

Enter news text: The government announced a new economic policy today  
Result: REAL NEWS (P(real)=0.62)

Enter news text: Scientists say bananas unlock psychic powers  
Result: FAKE NEWS (P(fake)=0.59)

Enter news text: A startup claims it built a battery lasting 50 years  
Result: UNCERTAIN (P(fake)=0.48, P(real)=0.52)
