import os
import json

def process_single_object(lines):
    """
    解析一组多行的 JSON 对象，提取键值对内容，确保值中的引号等被正确处理。
    """
    obj = {}
    for line in lines[1:-1]:  # 排除开头的 `{` 和结尾的 `}`
        line = line.strip()
        if not line or ':' not in line:
            continue
        key_raw, val_raw = line.split(':', 1)
        key = key_raw.strip().strip('"')
        val = val_raw.strip()

        # 去掉末尾逗号
        if val.endswith(','):
            val = val[:-1].strip()

        # 保留双引号内容，但内部可能还有双引号等，所以使用 json.loads 解析字符串
        if val.startswith('"') and val.endswith('"'):
            try:
                value = json.loads(val)  # 这一步会正确处理转义的内容
            except json.JSONDecodeError:
                value = val[1:-1]  # 回退策略
        else:
            value = val

        obj[key] = value
    return obj

def process_file(file_path):
    """
    按行解析非标准 JSON 文件，提取多个 JSON 对象。
    """
    objects = []
    current = []
    inside = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if line.strip() == '{':
                inside = True
                current = [line]
            elif line.strip() == '}' and inside:
                current.append(line)
                obj = process_single_object(current)
                if obj:
                    objects.append(obj)
                inside = False
            elif inside:
                current.append(line)
    return objects

def process_all_json_in_dir(root_dir, output_file):
    """
    遍历所有子目录中的 .json 文件，解析并合并为一个标准 JSON 数组。
    """
    all_data = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.json'):
                path = os.path.join(subdir, file)
                print(f"📂 正在处理: {path}")
                try:
                    objs = process_file(path)
                    print(f"  ✔️ 提取到 {len(objs)} 个对象")
                    all_data.extend(objs)
                except Exception as e:
                    print(f"  ❌ 错误文件: {path} -- {e}")

    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(all_data, out, indent=2, ensure_ascii=False)
    print(f"✅ 全部完成，共写入 {len(all_data)} 个对象到 {output_file}")

# 替换为你自己的路径
input_root = r'data'
output_path = r'merged_output.json'

process_all_json_in_dir(input_root, output_path)
