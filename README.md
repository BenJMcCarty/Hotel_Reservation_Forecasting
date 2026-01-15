# Hotel Reservation Forecasting

## Project Overview
This project generates forecasts for hotel reservation data using time series analysis and statistical modeling.

## Project Structure
```
Hotel_Reservation_Forecasting/
│
├── data/
│   ├── raw/              # Original, immutable data from source
│   └── processed/        # Cleaned and transformed data (daily records)
│
├── notebooks/            # Jupyter notebooks for exploration and analysis
│
├── src/                  # Source code for the project
│
├── models/               # Trained and serialized models
│
├── reports/              # Generated analysis, figures, and reports
│
├── config/               # Configuration files
│
├── pyproject.toml        # Project dependencies and metadata
│
└── README.md            # Project documentation
```

## Data Pipeline
1. **Data Sourcing**: Download data from static web link
2. **Data Transformation**: Convert single reservation records to daily records based on arrival/departure dates
3. **Forecasting**: Apply statsmodels and pmdarima for time series forecasting

## Setup

### Install uv (if not already installed)
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install project dependencies
```bash
uv sync
```

### Run Python scripts with uv
```bash
uv run python src/data_loader.py
```

## Technologies
- Python
- pandas
- statsmodels
- pmdarima
- jupyter
