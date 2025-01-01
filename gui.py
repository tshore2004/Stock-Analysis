import logging
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date
import yfinance as yf
from matplotlib.figure import Figure
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import requests
from app import make_prediction, newsSentiment

class StockApp:

    def __init__(self, root):
        """
        Initialize the Stock Prediction App.

        Args:
            root (tk.Tk): The root window.
        """

        self.root = root
        self.root.title("Stock Prediction and Analysis")
        self.root.geometry("1200x800")

        self.style = ttk.Style()
        self.style.configure("TFrame", padding=10)
        self.style.configure("TButton", padding=5)
        self.style.configure("TLabel", padding=5, font=("Helvetica", 12))
        self.style.configure("TNotebook", tabposition='wn')
        
        self.stocks = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META', 'TSLA']
        self.ticker_df = pd.read_csv('C:/Users/tshor/OneDrive/Desktop/Stock Predictor/Ticker-Symbols.csv')
        self.time_period_var = tk.StringVar(value="5d")
        self.news_sentiment = None
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.graph_canvas = None
        self.status_var = tk.StringVar(value="Ready")

        self.create_widgets()

    def create_widgets(self):
        """
        Create and arrange widgets in the GUI.
        """

        self.create_prediction_tab()
        self.create_stock_info_tab()
        self.create_historical_data_tab()
        self.create_news_sentiment_tab()
        self.create_help_tab()

    def predict(self, event=None):
        """
        Start the stock prediction process in a separate thread.
        """
        stock = self.stock_symbol.get().strip()
        if not stock:
            messagebox.showerror("Input Error", "Please select a stock symbol or enter a company name.")
            return

        if not self.is_valid_ticker(stock):
            ticker = self.get_ticker_from_name(stock)
            if not ticker:
                messagebox.showerror("Input Error", "Invalid stock symbol or company name.")
                return
        else:
            ticker = stock

        self.status_var.set("Processing Prediction...")
        self.result_label.config(text="")
        self.progress["value"] = 0
        self.progress["maximum"] = 100
        self.root.update_idletasks()

        try:
            logging.info(f"Starting prediction for ticker: {ticker}")
            data = self.download_stock_data(ticker, start='2013-01-01', end=date.today().strftime('%Y-%m-%d'))
            if data.empty:
                raise ValueError("No data available for the selected ticker.")

            action, probability, sentiment, trend, last_close, prediction_today, headlines = make_prediction(ticker, self.update_progress)
            
            logging.info(f"Prediction completed. Action: {action}, Prediction: {prediction_today}")

            self.ai_report_label.config(text=action)
            self.sentiment_label.config(text=sentiment[0].capitalize() if isinstance(sentiment, list) else sentiment.capitalize())
            self.trend_label.config(text=trend)
            self.prediction_today_label.config(text=f"${prediction_today:.2f}")
            self.last_close_label.config(text=f"${last_close:.2f}")
            self.probability_label.config(text=f"{probability[0] if isinstance(probability, list) else probability:.4f}")

            # Fetch news sentiment
            try:
                trader = newsSentiment(symbol=ticker)
                headlines, probabilities, sentiments = trader.getNewsSentiment()

                # Update news tab with fetched news
                self.news_text.delete('1.0', tk.END)
                for headline, sentiment in zip(headlines, sentiments):
                    self.news_text.insert(tk.END, f"{headline}\nSentiment: {sentiment.capitalize()}\n\n")
            except Exception as news_error:
                logging.error(f"Error fetching news: {str(news_error)}")
                self.news_text.delete('1.0', tk.END)
                self.news_text.insert(tk.END, "Error fetching news. Please try again later.")

            self.plot_stock_data(ticker)
            self.display_historical_data(ticker)
            self.display_stock_info(ticker)

        except ValueError as ve:
            error_message = f"Data Error: {str(ve)}"
            logging.error(error_message)
            logging.error(f"ValueError traceback: {traceback.format_exc()}")
            messagebox.showerror("Data Error", error_message)
            self.update_labels_error_state("No Data")

        except Exception as e:
            error_message = f"Prediction Error: {str(e)}"
            logging.error(error_message)
            logging.error(f"Exception traceback: {traceback.format_exc()}")
            messagebox.showerror("Prediction Error", error_message)
            self.update_labels_error_state("Error")

        finally:
            self.progress["value"] = 100
            self.root.update_idletasks()
            self.status_var.set("Ready")
            
    def update_labels_error_state(self, error_text):
        self.ai_report_label.config(text=error_text)
        self.sentiment_label.config(text="N/A")
        self.trend_label.config(text="N/A")
        self.prediction_today_label.config(text="N/A")
        self.last_close_label.config(text="N/A")
        self.probability_label.config(text="N/A")
    
    def is_valid_ticker(self, ticker):
        """
        Check if the given ticker symbol is valid.

        Args:
            ticker (str): The ticker symbol to validate.

        Returns:
            bool: True if the ticker is valid, False otherwise.
        """

        try:
            data = yf.Ticker(ticker).info
            return 'symbol' in data
        except Exception as e:
            print(f"Ticker validation error: {e}")  # Debugging
            return False

    def get_ticker_from_name(self, name):
        """
        Get the ticker symbol for a given company name.

        Args:
            name (str): The company name.

        Returns:
            str: The ticker symbol if found, None otherwise.
        """

        try:
            with open('C:\\Users\\tshor\\OneDrive\\Desktop\\Stock Predictor\\Ticker-Symbols.csv', mode='r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['Name'].strip().lower() == name.strip().lower():
                        return row['Ticker']
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to read the CSV file: {str(e)}")
        return None

    def download_stock_data(self, ticker, start=None, end=None, period=None):
        """
        Download stock data for the given ticker and time period.

        Args:
            ticker (str): The ticker symbol.
            start (str, optional): The start date for the data (YYYY-MM-DD). Defaults to None.
            end (str, optional): The end date for the data (YYYY-MM-DD). Defaults to None.
            period (str, optional): The time period for the data (e.g., '5d', '1mo'). Defaults to None.

        Returns:
            pd.DataFrame: The downloaded stock data.
        """

        try:
            print(f"Downloading data for {ticker} with start={start}, end={end}, period={period}")  # Debugging
            if period:
                data = yf.download(ticker, period=period, end=end)
            else:
                data = yf.download(ticker, start=start, end=end)

            if data.empty:
                print(f"No data found for {ticker} with start={start}, end={end}, period={period}")  # Debugging
                raise ValueError(f"No data available for {ticker} for the specified period.")
        
            return data
        
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                messagebox.showerror("API Limit Error", "You have exceeded the API rate limit. Please wait and try again later.")
            else:
                messagebox.showerror("HTTP Error", f"HTTP error occurred: {str(http_err)}")
            return pd.DataFrame()  # Return empty DataFrame if rate limit is exceeded or other HTTP error occurs

        except Exception as e:
            raise ValueError(f"Failed to retrieve data for {ticker}: {str(e)}")

    def display_historical_data(self, stock):
        """
        Display historical stock data for the given ticker.

        Args:
            stock (str): The ticker symbol.
        """

        try:
            data = self.download_stock_data(stock, start='2013-01-01', end=date.today().strftime('%Y-%m-%d'))
            if data.empty:
                raise ValueError("No historical data available for the selected ticker.")
            self.data_text.delete('1.0', tk.END)
            self.data_text.insert(tk.END, data.to_string())
        except ValueError as ve:
            messagebox.showerror("Data Error", f"Data Error: {str(ve)}")
        except Exception as e:
            messagebox.showerror("Data Error", f"Failed to retrieve historical data: {str(e)}")

    def display_stock_info(self, stock):
        """
        Display stock information for the given ticker.

        Args:
            stock (str): The ticker symbol.
        """

        try:
            data = yf.Ticker(stock)
            info = data.info

            self.company_name_label.config(text=info.get('shortName', 'N/A'))
            self.sector_label.config(text=info.get('sector', 'N/A'))
            self.industry_label.config(text=info.get('industry', 'N/A'))
            self.website_label.config(text=info.get('website', 'N/A'))
            self.current_price_label.config(text=f"${info.get('currentPrice', 'N/A'):.2f}")

            high_52week = info.get('fiftyTwoWeekHigh', 'N/A')
            low_52week = info.get('fiftyTwoWeekLow', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            volume = info.get('volume', 'N/A')
            avg_volume = info.get('averageVolume', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            div_yield = info.get('dividendYield', 'N/A')

            self.high_52week_label.config(text=f"${float(high_52week):.2f}" if isinstance(high_52week, (int, float, str)) and high_52week != 'N/A' else 'N/A')
            self.low_52week_label.config(text=f"${float(low_52week):.2f}" if isinstance(low_52week, (int, float, str)) and low_52week != 'N/A' else 'N/A')
            self.market_cap_label.config(text=f"${int(market_cap):,}" if isinstance(market_cap, (int, float, str)) and market_cap != 'N/A' else 'N/A')
            self.volume_label.config(text=f"{int(volume):,}" if isinstance(volume, (int, float, str)) and volume != 'N/A' else 'N/A')
            self.avg_volume_label.config(text=f"{int(avg_volume):,}" if isinstance(avg_volume, (int, float, str)) and avg_volume != 'N/A' else 'N/A')
            self.pe_ratio_label.config(text=f"{float(pe_ratio):.2f}" if isinstance(pe_ratio, (int, float, str)) and pe_ratio != 'N/A' else 'N/A')
            self.div_yield_label.config(text=f"{float(div_yield) * 100:.2f}%" if isinstance(div_yield, (int, float, str)) and div_yield != 'N/A' else 'N/A')

            self.plot_stock_data(stock)  # Plot stock data when displaying stock info

        except Exception as e:
            messagebox.showerror("Info Error", f"Failed to retrieve stock info: {str(e)}")

    def update_progress(self, message):
        """
        Update the progress bar and progress message.

        Args:
            message (str): The progress message to display.
        """
        # Update progress bar value
        if message == "Initialization":
            self.progress["value"] = 0
        elif message == "Prediction complete":
            self.progress["value"] = 100
        else:
            self.progress["value"] += 10  # Example increment; adjust as needed

        # Update status message
        self.status_var.set(message)
        self.root.update_idletasks()

    def clear_results(self):
        """
        Clears stock data previously displayed on other tabs.
        """

        self.result_label.config(text="")
        self.status_var.set("Ready")
        self.news_text.delete('1.0', tk.END)  # Clear news sentiment text
        self.data_text.delete('1.0', tk.END)  # Clear historical data text
        self.clear_stock_info()  # Clear stock info labels
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()  # Clear graph canvas

    def clear_stock_info(self):
        """
        Clears stock info previously displayed.
        """

        labels = [
            self.company_name_label, self.current_price_label, self.high_52week_label,
            self.low_52week_label, self.market_cap_label, self.volume_label,
            self.avg_volume_label, self.pe_ratio_label, self.div_yield_label
        ]
        for label in labels:
            label.config(text="")

    def plot_stock_data(self, stock):
        """
        Plot stock data for the given ticker.

        Args:
            ticker (str): The ticker symbol.
        """

        try:
            time_period = self.time_period_var.get()
            data = self.download_stock_data(stock, period=time_period)
            if data.empty:
                raise ValueError(f"No data available for the selected ticker '{stock}' in the specified time period '{time_period}'.")

            # Create a new figure and axis for the plot
            fig = Figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            data['Close'].plot(ax=ax, title=f"{stock} Stock Price - {time_period}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.grid(True)

            # Clear old graph if it exists
            if self.graph_canvas:
                self.graph_canvas.get_tk_widget().destroy()  # Destroy the old canvas widget
                self.graph_canvas = None

            # Create new canvas and plot
            self.graph_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.graph_canvas.draw()
            self.graph_canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

        except ValueError as ve:
            messagebox.showerror("Data Error", f"Data Error: {str(ve)}")
        except Exception as e:
            messagebox.showerror("Plotting Error", f"An error occurred while plotting: {str(e)}")


    def create_prediction_tab(self):
        """
        Create the stock prediction tab.
        """

        prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(prediction_tab, text="Prediction")

        stock_label = ttk.Label(prediction_tab, text="Stock Symbol or Company Name:")
        stock_label.pack(pady=5)

        self.stock_symbol = ttk.Combobox(prediction_tab, values=self.stocks)
        self.stock_symbol.pack(pady=5)
        self.stock_symbol.set("AAPL")

        self.stock_symbol.bind('<Return>', self.predict)

        predict_button = ttk.Button(prediction_tab, text="Predict", command=self.predict)
        predict_button.pack(pady=5)

        clear_button = ttk.Button(prediction_tab, text="Clear Results", command=self.clear_results)
        clear_button.pack(pady=5)

        # Create a frame for the prediction results
        results_frame = ttk.LabelFrame(prediction_tab, text="Prediction Results", padding=(10, 5))
        results_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Create labels for displaying prediction results
        self.result_label = ttk.Label(results_frame, text="", style="TLabel")
        self.result_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(results_frame, text="AI Report:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ai_report_label = ttk.Label(results_frame, text="")
        self.ai_report_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(results_frame, text="Sentiment:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.sentiment_label = ttk.Label(results_frame, text="")
        self.sentiment_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(results_frame, text="Prediction Trend:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.trend_label = ttk.Label(results_frame, text="")
        self.trend_label.grid(row=4, column=1, sticky=tk.W, pady=2)

        ttk.Label(results_frame, text="Today's Prediction:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.prediction_today_label = ttk.Label(results_frame, text="")
        self.prediction_today_label.grid(row=3, column=1, sticky=tk.W, pady=2)

        ttk.Label(results_frame, text="Last Closing Price:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.last_close_label = ttk.Label(results_frame, text="")
        self.last_close_label.grid(row=5, column=1, sticky=tk.W, pady=2)

        ttk.Label(results_frame, text="Prediction Accuracy Probability:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.probability_label = ttk.Label(results_frame, text="")
        self.probability_label.grid(row=6, column=1, sticky=tk.W, pady=2)

        # # Frame for the horizontal line
        # line_frame = ttk.Frame(prediction_tab)
        # line_frame.pack(fill=tk.X, pady=(10, 0))

        # # Add the horizontal line
        # ttk.Separator(line_frame, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Frame for the status text and progress bar
        progress_frame = ttk.Frame(prediction_tab)
        progress_frame.pack(pady=5, fill=tk.X)

        # Status label aligned to the left
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, style="TLabel")
        status_label.pack(side=tk.LEFT, padx=(5, 20))

        # Progress bar aligned to the right with fixed width
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=(20, 5))

    def create_stock_info_tab(self):
        """
        Create the stock information tab.
        """

        stock_info_tab = ttk.Frame(self.notebook)
        self.notebook.add(stock_info_tab, text="Stock Info")

        # Company Information Frame
        company_info_frame = ttk.LabelFrame(stock_info_tab, text="Company Information", padding=(10, 5))
        company_info_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(company_info_frame, text="Company Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.company_name_label = ttk.Label(company_info_frame, text="")
        self.company_name_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(company_info_frame, text="Sector:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.sector_label = ttk.Label(company_info_frame, text="")
        self.sector_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(company_info_frame, text="Industry:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.industry_label = ttk.Label(company_info_frame, text="")
        self.industry_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(company_info_frame, text="Website:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.website_label = ttk.Label(company_info_frame, text="")
        self.website_label.grid(row=3, column=1, sticky=tk.W, pady=2)

        # Stock Performance Frame
        stock_performance_frame = ttk.LabelFrame(stock_info_tab, text="Stock Performance", padding=(10, 5))
        stock_performance_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ttk.Label(stock_performance_frame, text="Current Price:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.current_price_label = ttk.Label(stock_performance_frame, text="")
        self.current_price_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(stock_performance_frame, text="52-Week High:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.high_52week_label = ttk.Label(stock_performance_frame, text="")
        self.high_52week_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(stock_performance_frame, text="52-Week Low:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.low_52week_label = ttk.Label(stock_performance_frame, text="")
        self.low_52week_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(stock_performance_frame, text="Market Cap:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.market_cap_label = ttk.Label(stock_performance_frame, text="")
        self.market_cap_label.grid(row=3, column=1, sticky=tk.W, pady=2)

        # Financial Statistics Frame
        financial_statistics_frame = ttk.LabelFrame(stock_info_tab, text="Financial Statistics", padding=(10, 5))
        financial_statistics_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(financial_statistics_frame, text="P/E Ratio:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.pe_ratio_label = ttk.Label(financial_statistics_frame, text="")
        self.pe_ratio_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(financial_statistics_frame, text="Dividend Yield:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.div_yield_label = ttk.Label(financial_statistics_frame, text="")
        self.div_yield_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(financial_statistics_frame, text="Volume:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.volume_label = ttk.Label(financial_statistics_frame, text="")
        self.volume_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(financial_statistics_frame, text="Avg Volume:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.avg_volume_label = ttk.Label(financial_statistics_frame, text="")
        self.avg_volume_label.grid(row=3, column=1, sticky=tk.W, pady=2)

        # Graph Frame
        graph_frame = ttk.LabelFrame(stock_info_tab, text="Stock Price Graph", padding=(10, 5))
        graph_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        ttk.Label(graph_frame, text="Select Time Period:").pack(anchor=tk.W, pady=5)
        self.time_period_menu = ttk.Combobox(graph_frame, textvariable=self.time_period_var, values=["5d","1mo", "3mo", "6mo", "1y", "5y", "max"])
        self.time_period_menu.pack(anchor=tk.W, pady=5)
        self.time_period_menu.bind("<<ComboboxSelected>>", lambda e: self.plot_stock_data(self.stock_symbol.get().strip()))

        self.graph_canvas_frame = ttk.Frame(graph_frame)
        self.graph_canvas_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.graph_frame = graph_frame  # Assign graph_frame to self.graph_frame

        # Configure grid weights to expand properly
        stock_info_tab.grid_rowconfigure(0, weight=1)
        stock_info_tab.grid_rowconfigure(1, weight=1)
        stock_info_tab.grid_columnconfigure(0, weight=1)
        stock_info_tab.grid_columnconfigure(1, weight=2)


    def create_historical_data_tab(self):
        """
        Create the historical stock data tab.
        """

        historical_data_tab = ttk.Frame(self.notebook)
        self.notebook.add(historical_data_tab, text="Historical Data")

        self.data_text = tk.Text(historical_data_tab)
        self.data_text.pack(pady=10, fill=tk.BOTH, expand=True)

    def create_news_sentiment_tab(self):
        """
        Create the news sentiment tab.
        """

        news_sentiment_tab = ttk.Frame(self.notebook)
        self.notebook.add(news_sentiment_tab, text="News Sentiment")

        self.news_text = tk.Text(news_sentiment_tab)
        self.news_text.pack(pady=10, fill=tk.BOTH, expand=True)

    def create_help_tab(self):
        """
        Create the help tab.
        """

        help_tab = ttk.Frame(self.notebook)
        self.notebook.add(help_tab, text="Help")

        help_text = (
            "This application allows you to predict stock prices and analyze stock data.\n"
            "1. Select a stock symbol or enter a company name.\n"
            "2. Click 'Predict' to get the prediction and related data.\n"
            "3. View stock info, historical data, and news sentiment in their respective tabs.\n"
            "4. Clear results using the 'Clear Results' button.\n"
            "For more details, refer to the documentation."
        )
        help_label = ttk.Label(help_tab, text=help_text, wraplength=800, style="TLabel")
        help_label.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()

