import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def collect_json_files(root_dir):
    json_file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.json'):
                json_file_list.append(os.path.join(dirpath, fname))
    return json_file_list

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen3b-ins")

# 设置最大 token 长度
MAX_TOKEN_LENGTH = 14000

def extract_conversations(data_item, filepath, idx=None):
    if not all(k in data_item for k in ["system_prompt", "input_query", "answer"]):
        id_info = f"{filepath}" if idx is None else f"{filepath} [index {idx}]"
        print(f"⚠️ Missing required fields in {id_info}")
        return None

    system_prompt = data_item["system_prompt"].strip()
    input_query = data_item["input_query"].strip()
    reason = data_item.get("reason", "").strip()
    answer = data_item["answer"].strip()

    # 拼接所有文本用于 token 长度计算
    combined_text = system_prompt + input_query + reason + answer
    token_len = len(tokenizer.tokenize(combined_text))

    if token_len > MAX_TOKEN_LENGTH:
        id_info = f"{filepath}" if idx is None else f"{filepath} [index {idx}]"
        print(f"🚫 Skipping {id_info}, token length {token_len} exceeds limit {MAX_TOKEN_LENGTH}")
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_query},
    ]

    if reason:
        combined = f"<think>{reason}</think>\n<answer>{answer}</answer>"
    else:
        combined = f"<answer>{answer}</answer>"

    messages.append({
        "role": "assistant",
        "content": combined
    })

    return {"messages": messages}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='raw', help='Input JSON root directory')
    parser.add_argument('--local_dir', default='output', help='Output directory for Parquet files')
    parser.add_argument('--enable_val', action='store_true', help='Enable validation split')
    parser.add_argument('--val_size', type=float, default=0.01, help='Validation set size ratio')
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    json_files = collect_json_files(input_dir)
    if not json_files:
        print(f"⚠️ No JSON files found in {input_dir}")
        return

    conversations = []
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read {filepath}: {e}")
            continue

        # Case 1: list of objects
        if isinstance(data, list):
            for idx, item in enumerate(data):
                convo = extract_conversations(item, filepath, idx)
                if convo:
                    conversations.append(convo)
        # Case 2: single object
        elif isinstance(data, dict):
            convo = extract_conversations(data, filepath)
            if convo:
                conversations.append(convo)
        else:
            print(f"⚠️ Unexpected format in {filepath} (not list or dict)")

    if not conversations:
        print("⚠️ No valid conversations extracted.")
        return

    df = pd.DataFrame(conversations)

    if args.enable_val:
        train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=42)
        train_path = os.path.join(local_dir, 'train.parquet')
        val_path = os.path.join(local_dir, 'test.parquet')
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        print(f"✅ Train Size: {len(train_df)}")
        print(f"✅ Val Size: {len(val_df)}")
        print(f"📁 Data saved to {local_dir}")
    else:
        train_path = os.path.join(local_dir, 'train.parquet')
        df.to_parquet(train_path, index=False)
        print(f"✅ Total {len(df)} examples saved to {train_path}")

if __name__ == '__main__':
    main()
