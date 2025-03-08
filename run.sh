
python -u run.py \
  --root_path ./m4 \
  --seasonal_patterns 'Monthly' \
  --data m4 \
  --features M \
  --count 100 \
  --max_concurrency 100\
  --norm
  # --norm --use_y \