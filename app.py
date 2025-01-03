import logging
import traceback
import ta
import time
import numpy as np
import yfinance as yf
from joblib import Parallel, delayed
from datetime import date
from alpaca_trade_api import REST
from timedelta import Timedelta
from newsML import estimate_sentiment
from lumibot.strategies.strategy import Strategy
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import  KFold, ParameterSampler, train_test_split

class NeuralNetwork:

    logging.basicConfig(level=logging.WARNING)

    def __init__(self, input_size, hidden_size, output_size):              
        """
        Initialize the neural network with given parameters.

        Args:
            input_size (int): Number of input neurons.
            hidden_size (int): Number of hidden neurons.
            output_size (int): Number of output neurons.
        """

        self.input_size = input_size
        self.hidden_size = [hidden_size] if isinstance(hidden_size, int) else hidden_size
        self.output_size = output_size
        
        self.weights = []
        self.biases = []

       # Initialize weights and biases for all layers
        if isinstance(input_size, int):
            prev_size = input_size
        elif isinstance(input_size, (list, tuple)):
            prev_size = np.prod(input_size)
        else:
            raise ValueError("input_size must be an int, list, or tuple")
        
        # Initialize weights and biases for all layers
        for size in hidden_size:
            self.weights.append(np.random.randn(prev_size, size) * np.sqrt(2. / prev_size))
            self.biases.append(np.zeros((1, size)))
            prev_size = size
        
        self.weights.append(np.random.randn(prev_size, output_size) * np.sqrt(2. / prev_size))
        self.biases.append(np.zeros((1, output_size)))



    def relu(self, x):
        """
        Relu activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Relu activated output.
        """

        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        Derivative of the relu function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Relu derivative output.
        """  

        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """
        Perform a forward pass through the network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the network.
        """
        X = X.reshape(X.shape[0], -1)
        self.activations = [X]
        self.z_values = []
        for i, weight in enumerate(self.weights):
            z = np.dot(self.activations[-1], weight) + self.biases[i]
            logging.debug(f"Layer {i} - Z shape: {z.shape}")
            a = self.relu(z)
            logging.debug(f"Layer {i} - Activation shape: {a.shape}")
            self.z_values.append(z)
            self.activations.append(a)
        return self.activations[-1]
    
    def backward(self, X, y, output, learning_rate):
        """"
        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels.
            output (numpy.ndarray): Predicted output.
            learning_rate (float): Learning rate.
        """
        X = X.reshape(X.shape[0], -1)  # Flatten the input
        y = y.reshape(output.shape)

        deltas = [(output - y) * self.relu_derivative(output)]

        logging.debug(f"Delta output shape: {deltas[-1].shape}")

        for i in range(len(self.hidden_size), 0, -1):
            delta = deltas[-1].dot(self.weights[i].T) * self.relu_derivative(self.activations[i])
            logging.debug(f"Delta shape for layer {i}: {delta.shape}")
            deltas.append(delta)
        
        deltas.reverse()

        for i in range(len(self.weights)):
            delta = np.squeeze(deltas[i])
            activation = np.squeeze(self.activations[i])

            
            logging.debug(f"Weight shape for layer {i}: {self.weights[i].shape}")
            logging.debug(f"Activation shape for layer {i}: {activation.shape}")
            logging.debug(f"Delta shape for layer {i}: {delta.shape}")

            self.weights[i] -= self.activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] -= np.sum(deltas[i], axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs, learning_rate, progress_callback):
        """
        Train the neural network.

        Args:
            X (numpy.ndarray): Training data.
            y (numpy.ndarray): Training labels.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            progress_callback (callable): Callback function to update training progress.
        """
        if progress_callback is None:
            progress_callback = lambda x: None

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                
                logging.info(f'Epoch {epoch}, Loss: {loss}')
            progress_callback((epoch / epochs) * 100)
    
    def predict(self, X):
        """
        Predict the output for given input.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted output.
        """
        logging.info(f"Predict method input shape: {X.shape}")
        output = self.forward(X)
        logging.info(f"Predict method output shape: {output.shape}")
        return output
    
    def normalize(self, data):
        """
        Normalize the data to a range of 0 to 1.

        Args:
            data (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Normalized data.
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / std
        return normalized_data

    def denormalize(self, normalized_data, original_data):
        """
        Denormalize the data to its original range.

        Args:
            data (numpy.ndarray): Original data.
            normalized_value (numpy.ndarray): Normalized value.

        Returns:
            numpy.ndarray: Denormalized value.
        """
        logging.info(f"Denormalize input shape: {normalized_data.shape}")
        mean = np.mean(original_data, axis=0)
        std = np.std(original_data, axis=0)
        
        if np.isscalar(normalized_data):
            logging.info(f"Denormalize output shape: {normalized_data.shape}")
            return float(normalized_data * std + mean)
        else:
            logging.info(f"Denormalize output shape: {normalized_data.shape}")
            return normalized_data * std + mean
        

def create_sequences(data, seq_length):
    """
    Create sequences of data for training.

    Args:
        data (numpy.ndarray): Input data.
        seq_length (int): Length of each sequence.

    Returns:
        tuple: Sequences of input data (X) and corresponding labels (y).
    """
    sequences = []
    labels = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        labels.append(data[i, 0])  # Assuming the first column is the target
    return np.array(sequences), np.array(labels)

def prepare_data(stock, start, end):
    """
    Prepare the stock data for training and testing.

    Args:
        stock (str): Stock symbol.
        start (str): Start date for data retrieval.
        end (str): End date for data retrieval.

    Returns:
        tuple: Processed data, training and testing sets.
    """

    data = yf.download(stock, start, end)
    data.reset_index(inplace=True)
    data.dropna(inplace=True)

    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()

    data.dropna(inplace=True)
   
    seq_length = 60
    features = ['Close', 'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low']
    
    # Ensure all feature columns exist and have the same length
    # for feature in features:
    #     assert feature in data.columns, f"Feature {feature} not found in data"
    #     assert data[feature].notna().all(), f"Feature {feature} contains NaN values"

    feature_data = data[features].values
    input_size = (seq_length, len(features))
    nn = NeuralNetwork(input_size=input_size, hidden_size=[50], output_size=1)
    normalized_data = nn.normalize(feature_data)   

    X, y = create_sequences(normalized_data, seq_length)

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Not enough data to create sequences. Please check the sequence length or the data length.")

    split_ratio = 0.8
    split_index = int(X.shape[0] * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    y_train = y_train.reshape(-1, 1)  # Ensure y is 2D
    y_test = y_test.reshape(-1, 1)
    
    return data, X_train, X_test, y_train, y_test, features

def grid_search(X_train, y_train, X_val, y_val, n_splits=5):
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]

    param_distributions = {
        'hidden_size': [[50], [100]],
        'learning_rate': [0.001, 0.005, 0.01],
        'epochs': [500, 1000]
    }

    # Generate random parameter combinations
    param_sampler = list(ParameterSampler(param_distributions, n_iter=10, random_state=42))
    def train_with_early_stopping(nn, X, y, val_split, learning_rate, epochs, patience=5):
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            output = nn.forward(X)
            loss = np.mean((y - output) ** 2)

            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > patience:
                print(f"Early stopping at epoch {epoch} with loss {best_loss}")
                break

            nn.backward(X, y, output, learning_rate)

        return best_loss

    def evaluate_params(params):
        try:
            nn = NeuralNetwork(input_size=X_train.shape[1:], hidden_size=params['hidden_size'], output_size=1)
            val_split = train_test_split(np.arange(X_train_small.shape[0]), test_size=.2, random_state=42)
            loss = train_with_early_stopping(nn, X_train_small, y_train_small, val_split, params['learning_rate'], params['epochs'], patience=5)
            return loss, params
        except Exception as e:
            print(f"Error with params {params}: {e}")
            return np.inf, params
        
    
    results = Parallel(n_jobs=-1)(delayed(evaluate_params)(params) for params in param_sampler)
    
    best_score, best_params = min(results, key=lambda x: x[0])
    
    if best_params is None:
        raise ValueError("No best parameters found in grid search.")
    
    return best_params

def train_neural_network(X, y, hidden_size, learning_rate, epochs, n_splits = 5, progress_callback = None):
    """
    Train the neural network on the given data.

    Args:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Training labels.
        hidden_size (int): Number of hidden neurons.
        learning_rate (float): Learning rate.
        epochs (int): Number of epochs.
        n_splits (int): Number of splits for cross-validation.
        progress_callback (callable): Callback function to update training progress.

    Returns:
        NeuralNetwork: Trained neural network.
    """
    if progress_callback is None:
        progress_callback = lambda x: None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    input_size = X.shape[1:]
    output_size = 1
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    total_iterations = n_splits * epochs
    current_iteration = 0

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

       
        for epoch in range(epochs):
            output = nn.forward(X_train_fold)
            nn.backward(X_train_fold, y_train_fold, output, learning_rate)

            if epoch % 100 == 0:
                loss = np.mean((y_train_fold - output) ** 2)

            current_iteration += 1
            progress_percentage = (current_iteration / total_iterations) * 100
            progress_callback(progress_percentage)
                # if progress_callback:
                #     progress_callback(int((epoch / epochs) * 100))  # Update progress

                #     if not callable(progress_callback):
                #         raise ValueError("progress_callback must be a callable function")
    
    return nn

class newsSentiment(Strategy):
    def __init__(self, symbol: str):
        """
        Initialize the news sentiment strategy with given symbol.

        Args:
            symbol (str): Stock symbol.
        """

        API_KEY = "AK27FDR1T3OF6OM8D4XN"
        API_SECRET = "4pseo5hYgCzcOEIPU5lZJ9aj4CTfTWaJWtsfJLrl"
        BASE_URL = "https://api.alpaca.markets"
        
        self.symbol = symbol
        self.api = REST(API_KEY, API_SECRET, BASE_URL)
        self.initialize()

    def initialize(self):
        """
        Initialize the strategy settings.
        """

        self.sleeptime = "24H"
        self.last_trade = None

    def getDate(self):
        """
        Get the current date and the date three days ago.

        Returns:
            tuple: Today's date and the date three days ago.
        """

        today = date.today()
        three_days = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days.strftime('%Y-%m-%d')

    def getNewsSentiment(self):
        """
        Retrieve the news sentiment for the given stock symbol.

        Returns:
            tuple: Headlines, probabilities, and sentiments.
        """

        today, three_days = self.getDate()
        news = self.api.get_news(self.symbol, start=three_days, end=today)
        headlines = [ev.__dict__["_raw"]["headline"] for ev in news]

        probabilities = []
        sentiments = []

        for headline in headlines:
            probability, sentiment = estimate_sentiment(headline)
            probabilities.append(probability)
            sentiments.append(sentiment)
        
        return headlines, probabilities, sentiments

def calculate_accuracy_metrics(y_true, y_pred):
            """
            Calculate evaluation metrics for regression models.

            Args:
                y_true (numpy.ndarray): True target values.
                y_pred (numpy.ndarray): Predicted values.

            Returns:
                dict: Dictionary containing calculated metrics.
            """
            metrics = {}

            # Mean Absolute Percentage Error (MAPE)
            metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred) * 100  # Expressed as percentage

            # Root Mean Squared Error (RMSE)
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))

            # R² Score
            metrics['R2'] = r2_score(y_true, y_pred)

            # Custom Accuracy Metric: Percentage of Predictions Within a Tolerance
            tolerance = 0.10  # 10% tolerance
            within_tolerance = np.abs((y_true - y_pred) / y_true) <= tolerance
            metrics['Accuracy_Percentage'] = np.mean(within_tolerance) * 100

            return metrics

def make_prediction(stock, progress_callback = None):
    """
    Make a stock price prediction using neural network and news sentiment.

    Args:
        stock (str): Stock symbol.
        progress_callback (callable): Callback function to update progress.

    Returns:
        tuple: Action, probabilities, sentiments, trend, last close price, prediction for today, and headlines.
    """
    if progress_callback is None:
        progress_callback = lambda x: None

    logging.info("Starting make_prediction function")

    start_time = time.time()

    try:
        progress_callback(10)

        # Data preparation
        logging.info("Preparing data")
        start = '2013-01-01'
        end = date.today()
        data, X_train, X_test, y_train, y_test, features = prepare_data(stock, start, end)
        logging.info(f"Data prepared. Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
        progress_callback(30)

        # Grid search
        logging.info("Starting grid search")
        split_ratio = 0.8
        split_index = int(len(X_train) * split_ratio)
        X_train, X_val = X_train[:split_index], X_train[split_index:]
        y_train, y_val = y_train[:split_index], y_train[split_index:]
        best_params = grid_search(X_train, y_train, X_val, y_val)
        logging.info(f"Best parameters found: {best_params}")
        progress_callback(50)

        # Neural network training
        logging.info("Training neural network")
        nn = train_neural_network(
            X_train, y_train, 
            hidden_size=best_params['hidden_size'], 
            learning_rate=best_params['learning_rate'], 
            epochs=best_params['epochs'], 
            progress_callback=progress_callback
        )
        logging.info("Neural network training completed")

        # Predictions
        logging.info("Making predictions")
        nn_predictions = nn.predict(X_test)
        logging.info(f"nn_predictions shape: {nn_predictions.shape}")

        nn_predictions = nn.denormalize(nn_predictions, data[features].values)
        y_test_denormalized = nn.denormalize(y_test, data[features].values)
        progress_callback(90) 

        # Evaluation metrics
        metrics = calculate_accuracy_metrics(y_test_denormalized, nn_predictions)
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.2f}")
        accuracy_percentage = metrics['Accuracy_Percentage']

        # Today's prediction
        logging.info("Making today's prediction")
        last_sequence = nn.normalize(data[features].values[-60:])
        last_sequence = last_sequence.reshape(1, 60, 8)
        prediction_today = nn.predict(last_sequence)
        logging.info(f"Raw prediction_today: type={type(prediction_today)}, shape={prediction_today.shape if hasattr(prediction_today, 'shape') else 'N/A'}, value={prediction_today}")


        if isinstance(prediction_today, np.ndarray):
            prediction_today = prediction_today.item() if prediction_today.size == 1 else prediction_today.flatten()[0]
        prediction_today = float(prediction_today)  # Ensure it's a Python float
        logging.info(f"Scalar prediction_today: type={type(prediction_today)}, value={prediction_today}")

        # Denormalize the prediction
        prediction_today = nn.denormalize(np.array([[prediction_today]]), data[features].values)
        if isinstance(prediction_today, np.ndarray):
            prediction_today = prediction_today.item() if prediction_today.size == 1 else prediction_today.flatten()[0]
        prediction_today = float(prediction_today)  # Ensure it's a Python float
        logging.info(f"Final prediction_today: type={type(prediction_today)}, value={prediction_today}")

        last_close = float(data['Close'].values[-1])
        logging.info(f"Last close: type={type(last_close)}, value={last_close}")
        
        # News sentiment
        logging.info("Fetching news sentiment")
        trader = newsSentiment(symbol=stock)
        headlines, probabilities, sentiments = trader.getNewsSentiment()

        # Determine trend and action
        logging.info("Determining trend and action")
        trend = ""
        action = ""

        price_increase = np.all(prediction_today > last_close)
        price_decrease = np.all(prediction_today < last_close)

        if price_increase and "positive" in sentiments and any(p > 0.999 for p in probabilities):
            trend = "The stock is likely to go up today."
            action = "Strong Buy"
        elif price_increase:
            trend = "The stock is predicted to go up today based on the data, the news is reflecting a neutral bias."
            action = "Likely Buy"
        elif "positive" in sentiments and any(p > 0.999 for p in probabilities):
            trend = "The stock is predicted to go up today based on the news."
            action = "Likely Buy"
        elif price_decrease and "negative" in sentiments and any(p > 0.999 for p in probabilities):
            trend = "The stock is likely to go down today."
            action = "Strong Sell"
        elif price_decrease:
            trend = "The stock is predicted to go down today based on the data, the news is reflecting a neutral bias."
            action = "Likely Sell"
        elif "negative" in sentiments and any(p > 0.999 for p in probabilities):
            trend = "The stock is predicted to go down today based on the news."
            action = "Likely Sell"
        else:
            trend = "The stock's movement is uncertain based on current data and news."
            action = "Hold"

        logging.info(f"Determined trend: {trend}")
        logging.info(f"Determined action: {action}")
        logging.info("Prediction process completed successfully")
        
        # timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")
        progress_callback(100) 
        return action, accuracy_percentage, sentiments, trend, last_close, prediction_today, headlines
    
    except Exception as e:
        logging.error(f"An error occurred in make_prediction: {str(e)}")
        logging.error(f"Error traceback: {traceback.format_exc()}")
        progress_callback(100)
        raise
