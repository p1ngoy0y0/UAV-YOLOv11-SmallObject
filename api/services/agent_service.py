import os
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# ==========================================
# 定義輸出格式
# ==========================================
class ObjectAnalysis(BaseModel):
    density: str = Field(description="人群密集度，必須是 LOW, MEDIUM 或 HIGH")
    flight_height: str = Field(description="拍攝高度，判斷為高空或低空兩種")

# ==========================================
# 本地 AI Agent 服務
# ==========================================
class LocalAgentService:
    def __init__(self):
        """
        透過多路徑探針機制，動態鎖定 Host 端的 Ollama 服務。
        """
        candidates = [
            os.getenv("OLLAMA_BASE_URL"),
            "http://host.docker.internal:11434",
            "http://172.17.0.1:11434",
            "http://172.18.0.1:11434",
            "http://localhost:11434"
        ]
        
        chosen_url = None
        
        for url in filter(None, candidates):
            try:
                response = httpx.get(url, timeout=0.5)
                if response.status_code == 200:
                    chosen_url = url
                    break
            except (httpx.ConnectError, httpx.ConnectTimeout):
                continue
        
        if not chosen_url:
            raise ConnectionError("🔴 [AIOps] 無法連接到任何 Ollama 服務，請檢查 OLLAMA_BASE_URL 環境變數或 Docker 網路設定。")

        self.local_client = AsyncOpenAI(
            base_url=f"{chosen_url}/v1",
            api_key="ollama" # Ollama 不需要 API Key
        )
        self.embedding_model = "all-minilm"
        self.analysis_model = "llava:latest" # 假設使用 llava 模型進行分析
        print(f"✅ [AIOps] LLM Agent 成功對接至通訊隧道: {chosen_url}")

    async def get_log_vector(self, raw_log_line: str):
        response = await self.local_client.embeddings.create(
            model=self.embedding_model,
            input=raw_log_line
        )
        return response.data[0].embedding

    async def analyze_yolo_features(self, yolo_features: str) -> dict:
        payload = {
            "model": self.analysis_model,
            "response_format": { "type": "json_object" }, 
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一個資深的無人機監控畫面分析師。"
                        "請依據 YOLO 偵測的 JSON 資料，分析畫面的情況。"
                        "回傳符合規格的 JSON: {'density': 'LOW/MEDIUM/HIGH', 'flight_height': 'HIGH/LOW'}"
                    )  
                },
                {
                    "role": "user",
                    "content": yolo_features 
                }
            ],
            "temperature": 0.0
        }
        
        try:
            response = await self.local_client.chat.completions.create(**payload)
            raw_json_str = response.choices[0].message.content
            return ObjectAnalysis.model_validate_json(raw_json_str).model_dump()
        except Exception as e:
            print(f"⚠️ [防禦攔截] 本地 LLM 分析管道發生異常: {str(e)}，啟動安全防禦降級...")
            return {"density": "UNKNOWN", "flight_height": "UNKNOWN"}
