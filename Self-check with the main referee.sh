PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 TRANSFORMERS_VERBOSITY=error \
python -u -m supplementor.cli \
  --engine qwen_vl \
  --model_id /mnt/BridgeLang/Qwen2.5-VL-3B-Instruct \
  --with_verify \
  --verify_model_id /mnt/BridgeLang/Qwen2.5-VL-72B-Instruct \
  --offline \
  --images test.jpg \
  --task "What action should the robot take to grasp the snack bag?" \
  --max_tokens 120 --print_vlog \
  1>/tmp/supp_out.json 2>/tmp/supp_err.log

python scripts/check_with_verify.py < /tmp/supp_out.json
