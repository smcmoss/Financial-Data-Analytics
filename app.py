import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import dash_bootstrap_components as dbc
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Simulate stock data
def simulate_stock_data(ticker, start='2022-01-01', end='2023-01-01'):
    np.random.seed(abs(hash(ticker)) % (10 ** 8))
    dates = pd.date_range(start=start, end=end, freq='B')
    prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    data = pd.DataFrame({'Date': dates, 'Close': prices})
    data.set_index('Date', inplace=True)
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

# Generate mock sentiment data
def mock_sentiment_data(dates):
    np.random.seed(42)
    return pd.Series(np.random.uniform(-1, 1, len(dates)), index=dates)

# Prepare features for model
def prepare_features(df, sentiment):
    df['Sentiment'] = sentiment
    df['Lag_1'] = df['Return'].shift(1)
    df['Lag_2'] = df['Return'].shift(2)
    df.dropna(inplace=True)
    X = df[['Lag_1', 'Lag_2', 'Sentiment']]
    y = df['Return']
    return X, y, df

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test

# Build app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Stock Return Prediction Dashboard"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Enter Stock Ticker (e.g., AAPL):"),
            dcc.Input(id='ticker-input', type='text', value='AAPL', debounce=True),
            html.Br(),
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker',
                start_date='2022-01-01',
                end_date='2023-01-01',
                display_format='YYYY-MM-DD'
            )
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='return-graph'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='error-message', style={'color': 'red'}))
    ])
], fluid=True)

@app.callback(
    Output('return-graph', 'figure'),
    Output('error-message', 'children'),
    Input('ticker-input', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_graph(ticker, start_date, end_date):
    try:
        stock_data = simulate_stock_data(ticker, start=start_date, end=end_date)
        sentiment_series = mock_sentiment_data(stock_data.index)
        X, y, processed_df = prepare_features(stock_data.copy(), sentiment_series)
        preds, actuals = train_model(X, y)

        df_plot = processed_df.iloc[-len(actuals):].copy()
        df_plot['Actual'] = actuals.values
        df_plot['Predicted'] = preds

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Actual'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Predicted'], mode='lines', name='Predicted'))
        fig.update_layout(title=f'Simulated vs Predicted Returns for: {ticker.upper()}', xaxis_title='Date', yaxis_title='Return')
        return fig, ""
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title='Error Occurred', xaxis_title='Date', yaxis_title='Return')
        return fig, f"An error occurred while processing data for '{ticker.upper()}'."

if __name__ == '__main__':
    app.run(debug=False)
