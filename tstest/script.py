from data_factory import data_provider
from m4 import M4Meta
import torch
import instructions
import os
import json

class GroundTruthAdder:
    def __init__(self, args):
        self.args = args
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = 0
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
            
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def add_ground_truth_to_json(self, existing_json_path, save_path):
        # 加载已有json数据
        with open(existing_json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        
        # 建立一个 input_query -> entry 映射
        input_query_to_entry = {entry["input_query"]: entry for entry in existing_data}
        updated_entries = []

        # 加载 dataloader
        train_data, train_loader = self._get_data(flag='train')

        count = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            if self.args.norm:
                means = batch_x.mean(1, keepdim=True).detach()
                batch_x = batch_x - means
                stdev = torch.sqrt(
                    torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                batch_x /= stdev
                batch_y = batch_y - means
                batch_y = batch_y / stdev

            batch_x = torch.reshape(batch_x, (batch_x.shape[0], batch_x.shape[2], -1)) #(B,1,L)
            batch_y = torch.reshape(batch_y, (batch_y.shape[0], batch_y.shape[2], -1)) #(B,1,L)

            for j in range(batch_x.shape[0]):
                count += 1
                if count > self.args.count:
                    break

                sample_x = batch_x[j].squeeze().numpy()
                sample_y = batch_y[j].squeeze().numpy()

                x_string = ",".join(map(str, sample_x))
                y_string = ",".join(map(str, sample_y))

                # 构造 input_query
                if self.args.use_y:
                    if self.args.norm:
                        input_query = instructions.use_norm_use_y.format(
                            history_len=self.args.seq_len,
                            future_len=self.args.pred_len,
                            history=x_string,
                            gt=y_string
                        )
                    else:
                        input_query = instructions.use_y.format(
                            history_len=self.args.seq_len,
                            future_len=self.args.pred_len,
                            history=x_string,
                            gt=y_string
                        )
                else:
                    if self.args.norm:
                        input_query = instructions.use_norm.format(
                            history_len=self.args.seq_len,
                            future_len=self.args.pred_len,
                            history=x_string
                        )
                    else:
                        input_query = instructions.nonorm_noy2.format(
                            history_len=self.args.seq_len,
                            future_len=self.args.pred_len,
                            history=x_string
                        )

                # 匹配并补充
                if input_query in input_query_to_entry:
                    input_query_to_entry[input_query]["ground_truth"] = '[future]' + y_string + '[/future]'
                    entry = input_query_to_entry[input_query]
                    updated_entries.append(entry)

        # 保存补充好的文件
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(updated_entries, f, ensure_ascii=False, indent=4)

        print(f"✅ 补充完成！共 {len(updated_entries)} 条添加了 ground_truth，已保存到：{save_path}")
