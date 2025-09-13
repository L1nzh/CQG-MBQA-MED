import json
import time
import openai
import requests

# ====== 配置 ======
# openai.api_key = "YOUR_OPENAI_API_KEY"  # 这里替换成你的 API Key
MODEL = "gpt-5-mini"
BATCH_INPUT_FILE = "batch_input.jsonl"
POLL_INTERVAL = 30  # 轮询间隔秒数

# ====== 1. 生成 batch 输入文件 ======
requests_data = [
    {
        "custom_id": "test_1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "用一句话介绍人工智能。"}
            ]
        }
    },
    {
        "custom_id": "test_2",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "用一句话介绍机器学习。"}
            ]
        }
    }
]

with open(BATCH_INPUT_FILE, "w", encoding="utf-8") as f:
    for req in requests_data:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")

print(f"✅ 生成输入文件: {BATCH_INPUT_FILE}")

# ====== 2. 上传输入文件 ======
with open(BATCH_INPUT_FILE, "rb") as f:
    uploaded_file = openai.files.create(file=f, purpose="batch")
print(f"✅ 上传完成: file_id={uploaded_file.id}")

# ====== 3. 创建 batch ======
batch = openai.batches.create(
    input_file_id=uploaded_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
print(f"✅ 创建 Batch: batch_id={batch.id}")

# ====== 4. 轮询状态直到完成 ======
while True:
    current = openai.batches.retrieve(batch.id)
    print(f"📡 Batch 状态: {current.status}")
    if current.status in ["completed", "failed", "cancelled", "expired"]:
        break
    time.sleep(POLL_INTERVAL)

# ====== 5. 下载结果文件 ======
if current.output_file_id:
    url = f"https://api.openai.com/v1/files/{current.output_file_id}/content"
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    response = requests.get(url, headers=headers)
    results = [json.loads(line) for line in response.text.strip().split("\n")]
    print("✅ 结果：")
    for r in results:
        print(r["custom_id"], "->", r.get("response", {}).get("body", {}))
else:
    print("❌ 没有结果文件，Batch 状态:", current.status)

# ====== 6. 如果有错误文件，下载查看 ======
if current.error_file_id:
    url = f"https://api.openai.com/v1/files/{current.error_file_id}/content"
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    response = requests.get(url, headers=headers)
    print("⚠️ 错误信息：")
    print(response.text)
