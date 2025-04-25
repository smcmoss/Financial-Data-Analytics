# Stock Sentiment Dashboard

This is a simulated stock return prediction dashboard built using Python and Dash. It allows users to:

- Input any stock ticker (simulated data)
- Select a date range
- View predicted vs. actual stock returns using a random forest model
- Visualize results interactively in a web dashboard

## Features

- Offline-compatible
- Sentiment analysis using mock data
- Predictive modeling with Random Forest
- Date range selector and ticker input

## Getting Started

### Prerequisites

You will need Python 3.8+ and `pip` installed.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-sentiment-dashboard.git
cd stock-sentiment-dashboard
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

4. Run the app:

```bash
python app.py
```

Visit [http://127.0.0.1:8050](http://127.0.0.1:8050) to view the dashboard.

## Notes

- All data is simulated to ensure offline compatibility.
- No financial advice implied.

## License

MIT
