
from huggingface_hub import snapshot_download

# 下载 Qwen2.5-3B-Instruct 的所有权重和配置文件
model_dir = snapshot_download(
    repo_id="Dear-Sloth/test2",
    revision="main",              # 或者填写特定 commit hash
    local_dir="./test2",     # 下载到的实际目录
    local_dir_use_symlinks=False # 禁止使用软链接，确保是物理文件
)

