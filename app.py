import os
import logging
import io
import base64
import gc
from datetime import date, datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Set 'Agg' backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_custom_objects

# Configure environment variables and settings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize Flask app
app = Flask(__name__, template_folder='template', static_url_path='/static', static_folder='static')
app.jinja_env.globals['enumerate'] = enumerate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load stock data
try:
    stock_data = pd.read_csv('static/symbols.csv')
    logger.info('Loaded stock symbols successfully.')
except Exception as e:
    logger.exception(f"Failed to load stock symbols: {e}")
    stock_data = pd.DataFrame(columns=['symbol', 'name'])

# Custom LSTM layer definition
def custom_lstm(**kwargs):
    kwargs.pop('time_major', None)
    return LSTM(**kwargs)

get_custom_objects().update({'LSTM': custom_lstm})

# Load LSTM model
try:
    model = load_model('keras_model.h5')
    logger.info('LSTM model loaded successfully.')
except Exception as e:
    logger.exception(f"Failed to load LSTM model: {e}")
    model = None

# Utility functions
def clear_memory(*vars):
    """Clear memory by deleting variables and running garbage collection."""
    for var in vars:
        try:
            del var
        except Exception as e:
            logger.error(f"Error deleting variable: {var}, {e}")
    gc.collect()

def format_large_numbers(number):
    """Format large numbers into human-readable format."""
    try:
        number = float(number)
    except (ValueError, TypeError):
        return 'N/A'
    
    if number >= 1_000_000_000:
        return f'{number / 1_000_000_000:.2f}B'
    elif number >= 1_000_000:
        return f'{number / 1_000_000:.2f}M'
    elif number >= 1_000:
        return f'{number / 1_000:.2f}K'
    return str(number)

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        base64_encoded = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        buf.close()
        plt.close(fig)
        return base64_encoded
    except Exception as e:
        logger.error(f"Error converting plot to base64: {e}")
        return None

def fetch_stock_details(symbol):
    """Fetch detailed stock information."""
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info

        # print(f"Stock Info for {symbol}: {stock_info}")  # Print the full data

        ex_dividend_timestamp = stock_info.get('exDividendDate')
        ex_dividend_date = datetime.fromtimestamp(ex_dividend_timestamp).strftime('%b %d, %Y') if ex_dividend_timestamp else 'N/A'

        dividend_yield = stock_info.get('dividendYield')
        forward_dividend_yield = f"{dividend_yield * 100:.2f}%" if dividend_yield else 'N/A'

        dividend_rate = stock_info.get('dividendRate')
        forward_dividend_yield_full = f"{dividend_rate:.2f} ({dividend_yield * 100:.2f}%)" if dividend_rate and dividend_yield else forward_dividend_yield

        return {
            "current_price": str(stock_info.get('currentPrice', 'N/A')),
            "previous_close": str(stock_info.get('previousClose', 'N/A')),
            "open": str(stock_info.get('open', 'N/A')),
            "bid": str(stock_info.get('bid', 'N/A')),
            "ask": str(stock_info.get('ask', 'N/A')),
            "days_range": f"{stock_info.get('dayLow', 'N/A')} - {stock_info.get('dayHigh', 'N/A')}",
            "volume": format_large_numbers(stock_info.get('volume', 0)),
            "market_cap": format_large_numbers(stock_info.get('marketCap', 0)),
            "52_week_range": f"{stock_info.get('fiftyTwoWeekLow', 'N/A')} - {stock_info.get('fiftyTwoWeekHigh', 'N/A')}",
            "1y_target_est": str(stock_info.get('targetMeanPrice', 'N/A')),
            "bookValue": str(stock_info.get('bookValue', 'N/A')),
            "ex_dividend_date": ex_dividend_date,
            "forward_dividend_yield": forward_dividend_yield_full
        }
    except Exception as e:
        logger.exception(f"Error fetching stock details for symbol {symbol}: {e}")
        return None

# Error handlers
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('500.html'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception(e)
    return render_template('error.html', message=str(e)), 500

# Routes
@app.route('/')
def home():
    logger.debug('Rendering home page')
    return render_template('index.html')

@app.route('/login')
def login():
    logger.debug('Rendering login page')
    return render_template('login.html')

@app.route('/stock_names', methods=['GET'])
def stock_names():
    """API endpoint to search for stock names."""
    try:
        query = request.args.get('query', '')
        if not query:
            return jsonify(stock_names=[])

        results = stock_data[
            stock_data['symbol'].str.contains(query, case=False, na=False) |
            stock_data['name'].str.contains(query, case=False, na=False)
        ]
        stock_list = [{'symbol': row['symbol'], 'name': row['name']} for _, row in results.iterrows()]
        return jsonify(stock_names=stock_list)
    except Exception as e:
        logger.exception(f"Error fetching stock names: {e}")
        return jsonify(stock_names=[]), 500

@app.route('/get_stock_details', methods=['GET'])
def get_stock_details():
    """API endpoint to get stock details."""
    try:
        symbol = request.args.get('symbol', '').strip().upper()
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400

        stock_details = fetch_stock_details(symbol)
        if stock_details is None:
            return jsonify({'error': 'Unable to fetch stock details'}), 500

        return jsonify(stock_details)
    except Exception as e:
        logger.exception("Error fetching stock details")
        return jsonify({'error': 'Unable to fetch stock details'}), 500

def format_major_holders(major_holders):
    """Format the major holders DataFrame for rendering."""
    try:
        if major_holders is not None and not major_holders.empty:
            major_holders_formatted = major_holders.copy()
            
            for column in major_holders_formatted.columns:
                if column == "institutionsCount":
                    # Display as an integer without formatting
                    major_holders_formatted[column] = major_holders_formatted[column].apply(
                        lambda x: int(x) if pd.notnull(x) else 'N/A'
                    )
                elif major_holders_formatted[column].dtype == float:
                    # Apply percentage formatting to other numeric columns
                    major_holders_formatted[column] = major_holders_formatted[column].apply(
                        lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else 'N/A'
                    )
            return major_holders_formatted
        else:
            return None
    except Exception as e:
        logger.error(f"Error formatting major holders: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    if model is None:
        logger.error("LSTM model is not loaded.")
        return render_template('error.html', message="Model not available. Please contact support."), 500

    try:
        today = date.today()
        end = today.strftime("%Y-%m-%d")
        start = '2014-01-01'

        user_input = request.form.get('stock-input', '').strip().upper()
        if not user_input:
            return render_template('error.html', message="No stock symbol provided.")

        logger.info(f"Fetching data for stock symbol: {user_input}")
        df = yf.download(user_input, start=start, end=end)

        if df.empty:
            logger.warning(f"No data found for stock symbol: {user_input}")
            return render_template('error.html', message="No data found for the given stock symbol.")

        # Get stock details and major holders
        stock = yf.Ticker(user_input)
        stock_info = stock.info
        stock_details = fetch_stock_details(user_input)

        # Process major holders data
        try:
            major_holders = stock.major_holders
            formatted_major_holders = format_major_holders(major_holders)
            if formatted_major_holders is not None:
                major_holders_html = formatted_major_holders.to_html(classes='table table-striped table-bordered')
            else:
                major_holders_html = "<p>No major holders found.</p>"
        except Exception as e:
            logger.error(f"Error retrieving major holders: {e}")
            major_holders_html = "<p>No major holders data available.</p>"

        # Generate plots
        description = df.describe().to_html(classes='table table-striped')

        # Plot 1: Closing Price
        fig1 = plt.figure(figsize=(11, 4.5))
        plt.plot(df['Close'], label='Close Price')
        plt.title('Closing Price vs Time Chart')
        plt.xlabel('Time')
        plt.ylabel('Closing Price')
        plt.legend()
        fig1_base64 = plot_to_base64(fig1)

        # Plot 2: Closing Price with 100 MA
        ma100 = df['Close'].rolling(window=100).mean()
        fig2 = plt.figure(figsize=(11, 4.5))
        plt.plot(ma100, 'r', label='100-Day MA')
        plt.plot(df['Close'], 'b', label='Close Price')
        plt.title('Closing Price vs Time Chart with 100-Day Moving Average')
        plt.xlabel('Time')
        plt.ylabel('Closing Price')
        plt.legend()
        fig2_base64 = plot_to_base64(fig2)

        # Plot 3: Closing Price with 100 MA and 200 MA
        ma200 = df['Close'].rolling(window=200).mean()
        fig3 = plt.figure(figsize=(11, 4.5))
        plt.plot(ma100, 'r', label='100-Day MA')
        plt.plot(ma200, 'g', label='200-Day MA')
        plt.plot(df['Close'], 'b', label='Close Price')
        plt.title('Closing Price vs Time Chart with 100 & 200-Day Moving Average')
        plt.xlabel('Time')
        plt.ylabel('Closing Price')
        plt.legend()
        fig3_base64 = plot_to_base64(fig3)

        # Prepare data for prediction
        df_processed = df.reset_index()
        columns_to_drop = ['Adj Close', 'Date']
        df_processed = df_processed.drop([col for col in columns_to_drop if col in df_processed.columns], axis=1, errors='ignore')

        if 'Close' not in df_processed.columns:
            logger.error("'Close' column not found in the DataFrame.")
            return render_template('error.html', message="'Close' column missing in data.")

        # Split and scale data
        data_training = pd.DataFrame(df_processed['Close'][0:int(len(df_processed) * 0.70)])
        data_testing = pd.DataFrame(df_processed['Close'][int(len(df_processed) * 0.70):])
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare test data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted.flatten() * scale_factor
        y_test = y_test * scale_factor

        # Plot 4: Original vs Predicted Prices
        fig4 = plt.figure(figsize=(11, 4.5))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')

        target_price = float(stock_info['targetMeanPrice']) if stock_info.get('targetMeanPrice') else y_predicted[-1]
        extended_x = list(range(len(y_predicted), len(y_predicted) + 252))
        extended_y = np.linspace(y_predicted[-1], target_price, 252)

        plt.plot(extended_x, extended_y, 'g--', label='1-Year Target Estimate')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        fig4_base64 = plot_to_base64(fig4)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_predicted)
        mse = mean_squared_error(y_test, y_predicted)
        with np.errstate(divide='ignore', invalid='ignore'):
            accuracy = 100 - np.mean(np.abs((y_test - y_predicted) / y_test)) * 100
            if np.isnan(accuracy):
                accuracy = 'N/A'

        # Clean up memory
        clear_memory(df, stock, data_training, data_testing, scaler, df_processed, 
                    final_df, input_data, x_test, y_test, y_predicted)

        return render_template('result.html',
                            description=description,
                            stock_info=stock_info,
                            fig1=fig1_base64,
                            fig2=fig2_base64,
                            fig3=fig3_base64,
                            fig4=fig4_base64,
                            mae=mae,
                            mse=mse,
                            accuracy=accuracy,
                            stock_details=stock_details,
                            major_holders=major_holders_html)

    except Exception as e:
        logger.exception(f"Error in prediction: {e}")
        return render_template('error.html', message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)