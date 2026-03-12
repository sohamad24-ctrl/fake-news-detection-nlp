# fake-news-detection-nlp
Fake News Detection using NLP, TF-IDF and Logistic Regression

## Project Overview
This project is a machine learning–based Fake News Detection system that classifies news text as **Real**, **Fake**, or **Uncertain** using Natural Language Processing techniques.

The model uses **TF-IDF vectorization** and **Logistic Regression**, followed by **probability calibration** to improve prediction reliability for short user inputs.

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

## Future Improvements
- Use transformer models such as BERT for improved accuracy
- Add multilingual support
- Build a web-based user interface
- Integrate real-time news verification APIs

## Project Structure

fake-news-detection-nlp/
│
├── train_model.py
├── train_calibration.py
├── predict_calibrated.py
├── short_inputs.csv
├── requirements.txt
└── README.md
