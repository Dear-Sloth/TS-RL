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