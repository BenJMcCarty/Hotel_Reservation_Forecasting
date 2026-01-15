"""
Forecasting module using statsmodels and pmdarima.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from typing import Tuple, Optional


def decompose_time_series(
    series: pd.Series,
    freq: int = 7,
    model: str = 'additive'
) -> dict:
    """
    Perform seasonal decomposition of time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    freq : int
        Seasonal period (default: 7 for weekly seasonality)
    model : str
        'additive' or 'multiplicative'
        
    Returns:
    --------
    dict
        Dictionary with trend, seasonal, and residual components
    """
    decomposition = seasonal_decompose(series, model=model, period=freq)
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'observed': decomposition.observed
    }


def fit_auto_arima(
    series: pd.Series,
    seasonal: bool = True,
    m: int = 7
) -> 'auto_arima':
    """
    Automatically fit ARIMA model using pmdarima.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    seasonal : bool
        Whether to fit seasonal ARIMA
    m : int
        Seasonal period
        
    Returns:
    --------
    Fitted auto_arima model
    """
    model = auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )
    
    return model


def forecast_future(
    model,
    n_periods: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate forecasts using fitted model.
    
    Parameters:
    -----------
    model : fitted model
        Fitted ARIMA model
    n_periods : int
        Number of periods to forecast
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Forecasted values and confidence intervals
    """
    forecast, conf_int = model.predict(
        n_periods=n_periods,
        return_conf_int=True
    )
    
    return forecast, conf_int


def evaluate_model(
    actual: pd.Series,
    predicted: pd.Series
) -> dict:
    """
    Calculate evaluation metrics for forecasting model.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    predicted : pd.Series
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
