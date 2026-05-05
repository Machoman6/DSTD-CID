from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('Qwen/Qwen3-Embedding-0.6B')
print("download:", model_dir)

