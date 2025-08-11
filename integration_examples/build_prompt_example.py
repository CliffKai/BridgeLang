import json, subprocess, sys
from pathlib import Path

def get_supplement(images, task):
    # 直接调用我们刚才的CLI；真实接入时你会改成Python内调用
    cmd = [
        sys.executable, "-m", "supplementor.cli",
        "--images", *images, "--task", task
    ]
    out = subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[1])
    j = json.loads(out.decode("utf-8"))
    return j["supplement"]

def build_openvla_prompt(task: str, supplement: str) -> str:
    return f"[TASK] {task}\n{supplement}"

if __name__ == "__main__":
    images = ["./demo.jpg"]   # 放一张任意图片
    task = "Pick up the red mug and place it on the top shelf."
    supplement = get_supplement(images, task)
    prompt = build_openvla_prompt(task, supplement)
    print("\n=== Final Prompt To OpenVLA ===\n", prompt)
