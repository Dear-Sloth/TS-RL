
python -u run.py \
  --root_path ./m4 \
  --seasonal_patterns 'Weekly' \
  --data m4 \
  --features M \
  --count 1000 \
  --max_concurrency 50

python -u run.py \
  --root_path ./m4 \
  --seasonal_patterns 'Daily' \
  --data m4 \
  --features M \
  --count 1000 \
  --max_concurrency 50

python -u run.py \
  --root_path ./m4 \
  --seasonal_patterns 'Yearly' \
  --data m4 \
  --features M \
  --count 2000 \
  --max_concurrency 50


python -u run.py \
  --root_path ./m4 \
  --seasonal_patterns 'Monthly' \
  --data m4 \
  --features M \
  --count 1000 \
  --max_concurrency 50


python -u run.py \
  --root_path ./m4 \
  --seasonal_patterns 'Quarterly' \
  --data m4 \
  --features M \
  --count 2000 \
  --max_concurrency 50


python -u run.py \
  --root_path ./m4 \
  --seasonal_patterns 'Hourly' \
  --data m4 \
  --features M \
  --count 1000 \
  --max_concurrency 50