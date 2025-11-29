from collections import OrderedDict
import numpy as np
import pandas as pd
import os
from m4 import M4Dataset, M4Meta

def group_values(values, groups, group_name):
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]], dtype=object)

def mase(forecast, insample, outsample, frequency):
    if len(insample) < 2 * frequency:
        return np.nan
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))

def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom

def mape(forecast, target):
    denom = np.abs(target)
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom

class M4Summary:
    def __init__(self, file_path, root_path):
        self.file_path = file_path
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv')

    def evaluate(self):
        grouped_owa = OrderedDict()

        naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)
        naive2_forecasts = np.array([v[~np.isnan(v)] for v in naive2_forecasts], dtype=object)

        model_mases, naive2_smapes, naive2_mases = {}, {}, {}
        grouped_smapes, grouped_mapes = {}, {}

        valid_groups = []

        for group_name in M4Meta.seasonal_patterns:
            file_name = os.path.join(self.file_path, group_name + "_forecast.csv")
            if not os.path.exists(file_name):
                print(f"⚠️ Forecast file not found for group: {group_name}, skipping")
                continue

            model_forecast = pd.read_csv(file_name).values.astype(np.float32)

            naive2_forecast = group_values(naive2_forecasts, self.test_set.groups, group_name)
            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)

            model_mases[group_name] = np.nanmean([
                mase(forecast=model_forecast[i], insample=insample[i], outsample=target[i], frequency=frequency)
                for i in range(len(model_forecast))])

            naive2_mases[group_name] = np.nanmean([
                mase(forecast=naive2_forecast[i], insample=insample[i], outsample=target[i], frequency=frequency)
                for i in range(len(model_forecast))])

            naive2_smapes[group_name] = np.nanmean(smape_2(naive2_forecast, target))
            grouped_smapes[group_name] = np.nanmean(smape_2(model_forecast, target))
            grouped_mapes[group_name] = np.nanmean(mape(model_forecast, target))

            valid_groups.append(group_name)

        grouped_smapes = self.summarize_groups(grouped_smapes, valid_groups)
        grouped_mapes = self.summarize_groups(grouped_mapes, valid_groups)
        grouped_model_mases = self.summarize_groups(model_mases, valid_groups)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes, valid_groups)
        grouped_naive2_mases = self.summarize_groups(naive2_mases, valid_groups)

        for k in grouped_model_mases.keys():
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                              grouped_smapes[k] / grouped_naive2_smapes[k]) / 2

        def round_all(d):
            return {k: np.round(v, 3) for k, v in d.items()}

        return round_all(grouped_smapes), round_all(grouped_owa), round_all(grouped_mapes), round_all(grouped_model_mases)

    def summarize_groups(self, scores, valid_groups):
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len(np.where(self.test_set.groups == group_name)[0])

        weighted_score = {}
        for g in ['Yearly', 'Quarterly', 'Monthly']:
            if g in scores:
                weighted_score[g] = scores[g] * group_count(g)
                scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            if g in scores:
                others_score += scores[g] * group_count(g)
                others_count += group_count(g)
                scores_summary[g] = scores[g]

        if others_count > 0:
            weighted_score['Others'] = others_score
            scores_summary['Others'] = others_score / others_count

        total_count = sum([group_count(g) for g in valid_groups])
        average = np.sum(list(weighted_score.values())) / total_count
        scores_summary['Average'] = average

        return scores_summary
