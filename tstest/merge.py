import os
import json

def merge_json_files(folder_path, output_path):
    merged_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 假设每个文件是一个字典或者列表
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)

    # 保存合并后的数据
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(merged_data, f_out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    folder = 'datas'
    output = 'merged.json'
    merge_json_files(folder, output)
