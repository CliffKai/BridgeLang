对，这个报错就是因为 **`hf download` 不支持 `--local-dir-use-symlinks` 和 `--resume`** 这两个老参数。把它们去掉就能跑了。给你一条可直接用的命令（保留镜像与加速）：

```bash
# 建议先装/升级依赖
pip install -U huggingface_hub hf_transfer

# 镜像 + 传输加速
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1   # 没装 hf_transfer 就取消或设为0

# 开始下（只保留有效参数）
hf download Qwen/Qwen2.5-VL-72B-Instruct \
  --local-dir ./Qwen2.5-VL-72B-Instruct \
  --include 'model-*.safetensors' 'model.safetensors.index.json' \
            'tokenizer*' '*config.json' \
            'chat_template.json' 'preprocessor_config.json' \
            'vocab.json' 'merges.txt'
# 如果仓库是 gated：命令最后加 --token hf_xxx 或先 `hf login`
```

小贴士

* `hf download` 会自动断点续传，不需要 `--resume`。
* “下载到本地目录”场景现在也不再用软链，所以没有 `--local-dir-use-symlinks`。
* 如果你没装 `hf_transfer`，就把 `HF_HUB_ENABLE_HF_TRANSFER` 关掉：

  ```bash
  unset HF_HUB_ENABLE_HF_TRANSFER   # 或 export HF_HUB_ENABLE_HF_TRANSFER=0
  ```

下载完成后，快速自检一下是否齐全（应有 38 个分片）：

```bash
cd ./Qwen2.5-VL-72B-Instruct
ls -1 model-*.safetensors | wc -l   # 期望输出：38
du -sh .                             # 看总体积是否接近百来GB
```

要不要我顺手给你一个“缺哪个分片就只补哪个”的小脚本？
