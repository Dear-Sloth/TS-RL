
python run_test.py \
  --root_path ./m4 \
  --seasonal_patterns Yearly \
  --data m4 \
  --features M \
  --model_path /data/dinghang/rl/ckpt_hf_year_3 \
  --batch_size 16 \
  --tensor_parallel_size 8
