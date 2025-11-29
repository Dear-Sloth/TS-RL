python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /data/dinghang/rl/checkpoints/gspo/gspo-ts-Llama-3.1-8B-Ins-RL/global_step_1011/actor \
    --target_dir /data/dinghang/rl/ckpt_hf_quarter3