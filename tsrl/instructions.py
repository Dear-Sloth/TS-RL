system=("You are an expert in time series forecasting. "
"Given a past time series, you can analyze it from various perspectives or extract meaningful features to make accurate predictions about future values. "
"You can consider trends, seasonality, anomalies, external factors, and all other relevant characteristics to provide reliable forecasts."
)

system_use_y=("You are an expert in time series forecasting. "
"Given a past time series and a future time series, you must pretend that you do not know the future values. "
"Instead, you can analyze the past time series from various perspectives or extracting meaningful features such as trends, seasonality, anomalies, external factors, and so on. "
"Based on this analysis, you generate a prediction that correctly matches the future time series while ensuring a well-reasoned and transparent forecasting process."
)
use_norm_use_y = (
    "Based on the following past time series of length {history_len}, please analyze and predict the future time series of length {future_len}. "
    "Your final prediction should be marked between: [future][future], with each number separated by a comma."
    "Here is the history time series: {history}"
    "Here is the future time series: {gt}"
)

use_norm = (
    "Based on the following past time series of length {history_len}, please infer and predict the future time series of length {future_len}."
    " Your final prediction should be marked between: [future][future], with each number separated by a comma."
    "Here is the history time series: {history}"    
)

use_y=(
    "Based on the following past time series of length {history_len}, please analyze and predict the future time series of length {future_len}. "
    "Your final prediction should be marked between: [future][future], with each number separated by a comma."
    "Here is the history time series: {history}"
    "Here is the future time series: {gt}"
)

basic_inp=(
    "Based on the following past time series of length {history_len}, please infer and predict the future time series of length {future_len}."
    " Your final prediction should be marked between: [future][future], with each number separated by a comma."
    "Here is the history time series: {history}"
)


system1=("You are an expert in time series forecasting. You do not rely on classical statistical methods such as Simple Moving Average,  Holt, ARIMA, or any other predefined models. Do not explicitly list mathematical equations or formulas in your response. Instead, you analyze the raw time series data directly, identifying meaningful trends, seasonality, turning points, or latent patterns. You are capable of reasoning based on structure, dynamics, or anomalies found in the data, and you generate accurate, data-driven forecasts without invoking named models.")
system2=("You are an expert in time series forecasting. Given a past time series, you analyze it by identifying meaningful patterns or features without using classical forecasting methods (e.g., Simple Moving Average, Holt's method). Do not explicitly list mathematical equations or formulas in your response. Clearly explain the reasoning or identified patterns behind your predictions before providing the forecast. Keep your reasoning concise and straightforward.")
system3=("You are an expert in time series forecasting. Given a past time series, you analyze it by identifying meaningful patterns or features without using classical forecasting methods (e.g., Simple Moving Average, Holt's method). Do not explicitly list mathematical equations or formulas in your response. Clearly explain the reasoning or identified patterns behind your predictions before providing the forecast. Do not excessively adjust or revise your analysis; perform one analysis and make one set of predictions directly.")

nonorm_noy1=("Based on the following past time series of length {history_len}, please infer and predict the future time series of length {future_len}. First, briefly explain the reasoning behind your prediction — what patterns or shifts you observed, and how you extrapolated them. Then provide the prediction, clearly marked between: [future][/future], with each number separated by a comma. Here is the history time series: {history}")
nonorm_noy2=("Based on the following past time series of length {history_len}, please first briefly explain your reasoning or any identified patterns, then infer and predict the future time series of length {future_len}. Your final prediction should be marked between: [future][/future], with each number separated by a comma. Here is the history time series: {history}")
nonorm_noy3=("Based on the following past time series of length {history_len}, please first briefly explain your reasoning or any identified patterns, then infer and predict the future time series of length {future_len}. Your final prediction should be marked between: [future][/future], with each number separated by a comma. Here is the history time series: {history}")


system_train=("You are an expert in time series forecasting. Given a past time series, you analyze it by identifying meaningful patterns or features without using classical forecasting methods (e.g., Simple Moving Average, Holt's method). Do not explicitly list mathematical equations or formulas in your response. Clearly explain the reasoning or identified patterns behind your predictions before providing the forecast. Keep your reasoning concise and straightforward.")

nonorm_noy_train = ("Based on the following past time series of length {history_len}, please first briefly explain your reasoning or any identified patterns, then infer and predict the future time series of length {future_len}. Your reasoning should be wrapped in <think></think>. Your final prediction should be wrapped in <future></future>, which must contain only the final predicted values, with each value separated by a comma. Here is the history time series: {history}")