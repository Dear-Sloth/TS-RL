
python -u run2.py \
  --root_path ./m4 \
  --seasonal_patterns 'Daily' \
  --data m4 \
  --features M \
  --count 1000 \
  --existing_json_path merged_output.json\
  --save_path 2.json 

python -u run2.py \
  --root_path ./m4 \
  --seasonal_patterns 'Yearly' \
  --data m4 \
  --features M \
  --count 2000 \
  --existing_json_path merged_output.json\
  --save_path 3.json 


python -u run2.py \
  --root_path ./m4 \
  --seasonal_patterns 'Monthly' \
  --data m4 \
  --features M \
  --count 1000 \
  --existing_json_path merged_output.json\
  --save_path 4.json 


python -u run2.py \
  --root_path ./m4 \
  --seasonal_patterns 'Quarterly' \
  --data m4 \
  --features M \
  --count 2000 \
  --existing_json_path merged_output.json\
  --save_path 5.json 


python -u run2.py \
  --root_path ./m4 \
  --seasonal_patterns 'Hourly' \
  --data m4 \
  --features M \
  --count 1000 \
  --existing_json_path merged_output.json\
  --save_path 6.json 