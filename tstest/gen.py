from data_factory import data_provider
from m4 import M4Meta
import torch
import instructions 
from call_llm import call_llm,async_call_llm
import os
import json
import asyncio
class data_generation:
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

    async def train_gen(self):
        train_data, train_loader = self._get_data(flag='train')
        count = 0
        path=f"./data/_{self.args.seasonal_patterns}/norm_{self.args.norm}_use_y_{self.args.use_y}_label_{self.args.label_len}_{self.args.seasonal_patterns}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tasks = []  
        max_concurrency = self.args.max_concurrency
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            if self.args.norm:
                means = batch_x.mean(1, keepdim=True).detach()
                batch_x = batch_x - means
                stdev = torch.sqrt(
                    torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                batch_x /= stdev
                batch_y = batch_y-means
                batch_y = batch_y/stdev
            batch_x = torch.reshape(batch_x, (batch_x.shape[0], batch_x.shape[2], -1)) #(B,1,L)
            batch_y = torch.reshape(batch_y, (batch_y.shape[0], batch_y.shape[2], -1)) #(B,1,L)
            
            for j in range(batch_x.shape[0]):
                count += 1
                if count > self.args.count:
                    break
                    
                sample_x = batch_x[j].squeeze().numpy()  # (L)
                sample_y = batch_y[j].squeeze().numpy()  # (L)
                x_string = ",".join(map(str, sample_x))
                y_string = ",".join(map(str, sample_y))
                
                if self.args.use_y:
                    sys_prompt = instructions.system_use_y
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
                    sys_prompt = instructions.system2
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
                
                # 为每个样本创建一个任务
                task = async_call_llm(sys_prompt, input_query)
                tasks.append((task, sys_prompt, input_query))
                
                # 当任务数量达到最大并发数时，执行这些任务
                if len(tasks) >= max_concurrency:
                    # 提取任务列表
                    task_list = [t[0] for t in tasks]
                    results = await asyncio.gather(*task_list)
                    
                    # 处理结果
                    for (reasons, anss), (_, sys_prompt, input_query) in zip(results, tasks):
                        if reasons and anss:
                            tmp = {
                                "system_prompt": sys_prompt,
                                "input_query": input_query,
                                "reason": reasons,
                                "answer": anss
                            }
                            with open(path, "a", encoding="utf-8") as f:
                                json.dump(tmp, f, ensure_ascii=False, indent=4)
                                f.write("\n")
                    
                    tasks.clear()  # 清空任务列表

        # 处理剩余的任务
        if tasks:
            task_list = [t[0] for t in tasks]
            results = await asyncio.gather(*task_list)
            
            for (reasons, anss), (_, sys_prompt, input_query) in zip(results, tasks):
                if reasons and anss:
                    tmp = {
                        "system_prompt": sys_prompt,
                        "input_query": input_query,
                        "reason": reasons,
                        "answer": anss
                    }
                    with open(path, "a", encoding="utf-8") as f:
                        json.dump(tmp, f, ensure_ascii=False, indent=4)
                        f.write("\n")
            