system = (
    "You are an expert in time series forecasting. "
    "Given a past time series, analyse it briefly before producing a forecast. "
    "Wrap your reasoning inside <think></think> and provide the final {future_len} values inside <future></future> "
    "with numbers separated by commas. Do not output any extra text after </future>."
)

system_use_y = (
    "You are an expert in time series forecasting. "
    "Given a past time series and its future continuation, imagine you do not know the future values. "
    "Analyse the history, wrap your reasoning in <think></think>, and output the forecast using <future></future> "
    "with comma-separated numbers. Do not reveal the provided future values directly."
)

use_norm_use_y = (
    "Based on the following past time series of length {history_len}, please analyse and predict the future series "
    "of length {future_len}. Wrap reasoning with <think></think> and place the {future_len} predictions inside "
    "<future></future> as comma-separated numbers. "
    "Here is the history time series: {history}"
    "Here is the future time series: {gt}"
)

use_norm = (
    "Based on the following past time series of length {history_len}, please infer and predict the future time series "
    "of length {future_len}. Wrap your reasoning inside <think></think> and output exactly {future_len} comma-separated "
    "values inside <future></future>. Here is the history time series: {history}"
)

use_y = (
    "Based on the following past time series of length {history_len}, please analyse and predict the future time series "
    "of length {future_len}. Wrap reasoning in <think></think> and output {future_len} comma-separated numbers inside "
    "<future></future>. Here is the history time series: {history}"
    "Here is the future time series: {gt}"
)

basic_inp = (
    "Based on the following past time series of length {history_len}, please infer and predict the future time series "
    "of length {future_len}. Wrap your reasoning inside <think></think> and provide exactly {future_len} comma-separated "
    "values within <future></future>. Here is the history time series: {history}"
)


system_train=("You are an expert in time series forecasting. Given a past time series, you analyze it by identifying meaningful patterns or features without using classical forecasting methods (e.g., Simple Moving Average, Holt's method). Do not explicitly list mathematical equations or formulas in your response. Clearly explain the reasoning or identified patterns behind your predictions before providing the forecast. Keep your reasoning concise and straightforward.")

nonorm_noy_train = ("Based on the following past time series of length {history_len}, please first briefly explain your reasoning or any identified patterns, then infer and predict the future time series of length {future_len}. Your reasoning should be wrapped in <think></think>. Your final prediction should be wrapped in <future></future>, which must contain only the final predicted values, with each value separated by a comma. Here is the history time series: {history}")