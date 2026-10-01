# Bank Statement Analyzer

A Flask-based web application that analyzes bank transaction statements, categorizes transactions using Machine Learning, and presents financial insights through a web dashboard.

## Features

* Upload bank statements in CSV format
* Automatically categorize transactions using Machine Learning
* Calculate total income and expenses
* Display current balance
* Identify highest income and expense transactions
* Show the most frequent transaction category
* Visualize income vs. expense using charts
* Search and filter transactions
* Download filtered transactions as CSV
* Store transaction data using SQLite

## Technologies Used

* **Python**
* **Flask** — Web application framework
* **Pandas** — Data processing
* **Scikit-learn** — Machine Learning
* **TF-IDF** — Text feature extraction
* **Multinomial Naive Bayes** — Transaction classification
* **SQLite** — Database
* **Matplotlib** — Data visualization
* **HTML/CSS** — Frontend

## Machine Learning

The application classifies transaction descriptions into spending categories.

The ML pipeline consists of:

1. Transaction descriptions are extracted from the training dataset.
2. Categories are normalized into broader groups.
3. TF-IDF converts transaction descriptions into numerical features.
4. A Multinomial Naive Bayes classifier is trained on the resulting features.
5. The trained model predicts categories for newly uploaded transactions.

### Model Performance

Using the current training configuration and an 80/20 train-test split with `random_state=42`,The model achieved 93.41% accuracy on an 80/20 random train-test split using the current dataset and random_state=42.

Model performance can vary depending on the dataset and train-test split.

## Project Structure

```text
bank-statement-analyzer/
│
├── app.py
├── database.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── database/
│
├── dataset/
│   └── transactions.csv
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── sample_data/
│   └── sample_statement.csv
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│       ├── bar_chart.png
│       └── pie_chart.png
│
└── templates/
    ├── home.html
    ├── upload.html
    ├── dashboard.html
    ├── transactions.html
    └── about.html
```

## CSV Format

The uploaded statement should contain the following columns:

```text
Date
Description
Debit
Credit
Balance
```


Example:

```text
Date,Description,Debit,Credit,Balance
01-07-2026,Salary,0,50000,50000
02-07-2026,Swiggy,450,0,49550
03-07-2026,Amazon,1200,0,48350
04-07-2026,Uber,350,0,48000
05-07-2026,Electricity Bill,2200,0,45800
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd bank-statement-analyzer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

The trained model files are included in the repository.

If you want to retrain the model:

```bash
python train_model.py
```

This generates:

```text
model/model.pkl
model/vectorizer.pkl
```

## Run the Application

Start the Flask application:

```bash
python app.py
```

Then open the local application in your browser at:

```text
http://127.0.0.1:5000
```

## Application Flow

```text
Upload CSV
    ↓
Validate Data
    ↓
TF-IDF Feature Extraction
    ↓
ML Category Prediction
    ↓
Store Transactions in SQLite
    ↓
Calculate Financial Insights
    ↓
Display Dashboard
```

## Future Improvements

* Support additional bank statement formats
* Add user authentication
* Add monthly and yearly spending trends
* Improve transaction categorization with a larger dataset
* Add category-wise spending charts
* Add automated anomaly detection
* Support PDF bank statements

## Disclaimer

This project is intended for educational and demonstration purposes. It should not be used as a replacement for official banking or financial software.
