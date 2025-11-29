import pandas as pd
import os

# 路径替换为你实际的结果目录
csv_path = 'Weekly_forecast.csv'

# 读取 CSV（包含错误的 index 列）
df = pd.read_csv(csv_path)

# 删除第一列（通常是 index 名为 'Unnamed: 0' 或其他）
df = df.iloc[:, 1:]

# 保存回原文件，不写 index
df.to_csv(csv_path, index=False)

print("✅ 已清理 index 列并保存至原文件，M4Summary 现在应该能正常运行。")
