import os
import httpx
from pydantic import BaseModel, Field

# ==========================================
# 定義文字規範
# ==========================================
class PeopleAnalysis(BaseModel):
    density: str = Field(description="人群密集度，必須是 LOW, MEDIUM 或 HIGH")
    flight_height: str = Field(description="拍攝高度，判斷為高空或低空兩種")

# ==========================================
# [核心管道] 雲端 LLM 異步調用函數
# ==========================================
async def call_cloud_llm_pipeline(yolo_features: str) -> dict:
    # 安全防線：從環境變數讀取 API Key
    api_key = os.getenv("OPENAI_API_KEY", "mock-key-for-test")
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 強迫啟用 json_object Mode
    payload = {
        "model": "gpt-4o-mini",
        "response_format": { "type": "json_object" }, 
        "messages": [
            {
                "role": "system",
                "content": "你是一個資深的監控畫面分析師。/n"
                "請依據 YOLO 偵測的資料，分析偵測畫面的情況，/n"
                "並回傳符合規格的 JSON: {\"density\": \"LOW/MEDIUM/HIGH\", \"flight_height\": HIGH/LOW}"  
            },
            {
                "role": "user",
                "content": yolo_features 
            }
        ],
        "temperature": 0.0 # 降到最低，確保結果穩定、不瞎編
    }
    
    # [方塊二] 利用 httpx 開啟非同步通道，控管 10 秒逾時（防禦性編程）
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status() # 如果 HTTP 狀態碼不是 200，立刻拋出異常
            
            # 解析回傳的原始 JSON 字串
            raw_json_str = response.json()["choices"][0]["message"]["content"]
                        
            # 驗證完全通關，安全回傳標準 Python 字典
            return PeopleAnalysis.model_validate_json(raw_json_str).model_dump()
            
        except Exception as e:
            # 安全降級機制（Fallback）：網路掛掉或大模型亂吐格式時，回傳安全預設值，大盤絕不卡死
            print(f"⚠️ [防禦攔截] 雲端管道發生異常: {str(e)}，啟動安全防禦降級...")
            return {"density": "UNKNOWN", "flight_height": "UNKNOWN"}