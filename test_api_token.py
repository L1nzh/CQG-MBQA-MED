#!/usr/bin/env python3
"""
使用 OpenAI 的 GPT-4o-mini 模型通过 Standard API 和 Batch API 生成问题。
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from openai.types import Batch
from openai import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
import dotenv

# 加载环境变量
dotenv.load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==================== 配置 ====================
MODEL_NAME = "gpt-4o-mini"  # 或 "gpt-4o-mini-2024-07-18"
MAX_RETRIES = 3
RETRY_DELAY = 1  # 秒
BATCH_TIMEOUT_CHECK = 30  # 检查 Batch 状态的间隔（秒）

# ==================== 数据结构 ====================

@dataclass
class QuestionRequest:
    """表示一个生成问题的请求"""
    text: str
    context: str = ""
    system_prompt: str = "You are a helpful assistant that generates insightful, binary (yes/no) questions about the given text."

    def to_openai_format(self, request_id: str) -> Dict[str, Any]:
        """转换为 OpenAI Batch API 所需的格式"""
        prompt = f"Text: {self.text}\nContext: {self.context}\n\nGenerate one insightful binary (yes/no) question about this text."
        return {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
        }

# ==================== 核心功能 ====================

def generate_question_standard(request: QuestionRequest) -> str:
    """
    使用 Standard API 实时生成一个问题（同步）。
    
    Args:
        request: 问题生成请求
        
    Returns:
        生成的问题文本
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": f"Text: {request.text}\nContext: {request.context}\n\nGenerate one insightful binary (yes/no) question about this text."}
                ],
                max_tokens=50,
                temperature=0.7
            )
            # 提取生成的问题
            question = response.choices[0].message.content.strip()
            logger.info(f"✅ Standard API: Generated question: {question}")
            return question

        except (APITimeoutError, APIConnectionError) as e:
            logger.warning(f"⚠️  Standard API Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))  # 指数退避
            else:
                logger.error("❌ Standard API: All attempts failed.")
                return "Error: Failed to generate question."
        except Exception as e:
            logger.error(f"❌ Standard API Unexpected error: {e}")
            return "Error: Unexpected error."

def create_batch_requests(requests: List[QuestionRequest]) -> List[Dict[str, Any]]:
    """
    将 QuestionRequest 列表转换为 OpenAI Batch API 所需的请求列表。
    
    Args:
        requests: 请求列表
        
    Returns:
        格式化后的请求列表
    """
    batch_requests = []
    for i, req in enumerate(requests):
        batch_req = req.to_openai_format(f"request-{i}")
        batch_requests.append(batch_req)
    return batch_requests

def upload_batch_file(requests_jsonl: str) -> str:
    """
    上传包含请求的 .jsonl 文件到 OpenAI。
    
    Args:
        requests_jsonl: .jsonl 文件的路径
        
    Returns:
        上传的文件 ID
    """
    try:
        file = client.files.create(
            file=open(requests_jsonl, "rb"),
            purpose="batch"
        )
        logger.info(f"📁 Batch File Uploaded. ID: {file.id}")
        return file.id
    except Exception as e:
        logger.error(f"❌ Failed to upload batch file: {e}")
        raise

def create_and_run_batch(file_id: str) -> Batch:
    """
    创建并启动一个 Batch 任务。
    
    Args:
        file_id: 上传的文件 ID
        
    Returns:
        Batch 对象
    """
    try:
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        logger.info(f"🚀 Batch Created. Batch ID: {batch.id}")
        return batch
    except Exception as e:
        logger.error(f"❌ Failed to create batch: {e}")
        raise

def monitor_batch(batch_id: str) -> Batch:
    """
    监控 Batch 的状态，直到完成或失败。
    
    Args:
        batch_id: Batch ID
        
    Returns:
        最终的 Batch 对象
    """
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            logger.info(f"📊 Batch Status: {batch.status} | "
                        f"Requests: {batch.request_counts.total} "
                        f"(Completed: {batch.request_counts.completed}, "
                        f"Failed: {batch.request_counts.failed})")
            
            if batch.status in ["completed", "failed", "expired", "cancelled"]:
                return batch
            
            time.sleep(BATCH_TIMEOUT_CHECK)
            
        except Exception as e:
            logger.error(f"❌ Error monitoring batch: {e}")
            time.sleep(BATCH_TIMEOUT_CHECK)

def download_batch_results(batch: Batch) -> List[Dict[str, Any]]:
    """
    下载 Batch 的输出结果。
    
    Args:
        batch: Batch 对象
        
    Returns:
        解析后的结果列表
    """
    if not batch.output_file_id:
        logger.warning("⚠️  No output file for this batch.")
        return []
    
    try:
        # 下载输出文件内容
        content = client.files.content(batch.output_file_id)
        output_path = f"batch_output_{batch.id}.jsonl"
        with open(output_path, 'w') as f:
            f.write(content.text)
        logger.info(f"📥 Batch output saved to {output_path}")
        
        # 解析结果
        results = []
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # 提取问题
                    question = item['response']['body']['choices'][0]['message']['content'].strip()
                    results.append({
                        "id": item['custom_id'],
                        "question": question
                    })
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to download or parse batch results: {e}")
        return []

# ==================== 主函数示例 ====================

def main():
    """演示 Standard API 和 Batch API 的使用"""
    
    # 示例文本列表（在 CQG-MBQA 中，这些可能是聚类的中心句或代表性句子）
    sample_texts = [
        "The cat sat on the mat.",
        "Machine learning models can learn from data.",
        "Photosynthesis converts sunlight into chemical energy."
    ]
    
    # 1. 使用 Standard API 生成问题（实时）
    print("\n" + "="*60)
    print("🟢 USING STANDARD API (Real-time)")
    print("="*60)
    
    standard_questions = []
    for text in sample_texts:
        req = QuestionRequest(text=text, context="General knowledge.")
        question = generate_question_standard(req)
        standard_questions.append(question)
    
    print("\nGenerated Questions (Standard API):")
    for q in standard_questions:
        print(f"  - {q}")
    
    # 2. 使用 Batch API 生成问题（异步）
    print("\n" + "="*60)
    print("🔵 USING BATCH API (Asynchronous)")
    print("="*60)
    
    # 创建请求
    batch_requests = [QuestionRequest(text=text, context="General knowledge.") for text in sample_texts]
    formatted_requests = create_batch_requests(batch_requests)
    
    # 保存为 .jsonl 文件
    input_file = "batch_input.jsonl"
    with open(input_file, 'w') as f:
        for req in formatted_requests:
            f.write(json.dumps(req) + '\n')
    logger.info(f"📄 Batch input saved to {input_file}")
    
    # 上传文件
    file_id = upload_batch_file(input_file)
    
    # 创建并运行 Batch
    batch = create_and_run_batch(file_id)
    
    # 监控 Batch 状态
    print("\n⏳ Monitoring batch status... (This may take a few minutes)")
    final_batch = monitor_batch(batch.id)
    
    if final_batch.status == "completed":
        # 下载并解析结果
        batch_results = download_batch_results(final_batch)
        print(f"\n✅ Batch completed! Retrieved {len(batch_results)} results.")
        
        print("\nGenerated Questions (Batch API):")
        for result in batch_results:
            print(f"  - {result['question']}")
    else:
        logger.error(f"❌ Batch failed with status: {final_batch.status}")

if __name__ == "__main__":
    main()