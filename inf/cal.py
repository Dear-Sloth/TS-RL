import os
from utils.m4_summary import M4Summary
folder_path_m4 = './m4_results/' + 'test2' + '/'
root_path='./m4'
if 'Weekly_forecast.csv' in os.listdir(folder_path_m4) :
                m4_summary = M4Summary(folder_path_m4, root_path)
                smape_results, owa_results, mape_results, mase_results = m4_summary.evaluate()
                print('smape:', smape_results)
                print('mape:', mape_results)
                print('mase:', mase_results)
                print('owa:', owa_results)
else:
                print('After all 6 tasks are finished, you can calculate the averaged index')

