# Stock Predictor App


## Overview
The Stock Predictor App is an advanced tool for forecasting stock prices and analyzing market data. Combining technical analysis, machine learning models, and sentiment analysis from news, the app provides actionable insights to help users make informed trading decisions.

## Features
- **Stock Prediction:** Predict future stock prices using machine learning models.
- **News Sentiment Analysis:** Evaluate market sentiment based on financial news.
- **Interactive GUI:** User-friendly interface with real-time stock data, prediction results, and visualizations.
- **Comprehensive Stock Data:** View historical data, company information, and stock trends.

## Requirements
To run the Stock Predictor App, ensure you have the following installed:
- Python 3.8 or later
- Required Python packages (see `requirements.txt`):
  - `numpy`
  - `pandas`
  - `yfinance`
  - `matplotlib`
  - `tkinter`
  - `transformers`
  - `torch`
  - `ta`
  - `scikit-learn`
  - `joblib`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/stock-predictor.git
   cd stock-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python gui.py
   ```

## Usage
1. Launch the app by running `gui.py`.
2. Enter a stock symbol or company name in the Prediction tab.
3. Click "Predict" to generate a forecast and analyze sentiment.
4. Explore tabs for historical data, company information, and news sentiment.

## File Structure
```
.
├── app.py              # Core logic for stock prediction
├── gui.py              # Graphical user interface implementation
├── newsML.py           # News sentiment analysis module
├── requirements.txt    # Python dependencies
├── LICENSE             # License details
└── README.md           # Project documentation
```

## Key Components
### 1. Stock Prediction
- **Data Preparation:** Fetches and processes stock data from Yahoo Finance.
- **Neural Network:** Custom implementation with hyperparameter tuning using grid search.

### 2. News Sentiment Analysis
- Utilizes FinBERT to classify news articles into positive, negative, or neutral sentiments.

### 3. Graphical Interface
- Developed using `tkinter`.
- Features multiple tabs for predictions, stock info, historical data, and news sentiment.

## Example Workflow
1. **Prediction:**
   - Input: Stock ticker (e.g., `AAPL` for Apple).
   - Output: Predicted stock price, trend, and actionable advice (e.g., "Strong Buy").

2. **Sentiment Analysis:**
   - Input: Recent news headlines.
   - Output: Sentiment classification with probabilities.

3. **Visualization:**
   - Displays historical stock trends and moving averages.

## Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request. For major changes, open an issue to discuss what you'd like to change.
