import json
from fastapi import APIRouter, Request, status, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

router = APIRouter(prefix="/agent", tags=["AIOps Agent"])

# ==========================================
# API Payload & Response 定義 (Enforcing SoC)
# ==========================================
class LogPayload(BaseModel):
    server_id: int
    error_msg: str

class VectorTriageResponse(BaseModel):
    status: str = "success"
    server_id: int
    raw_content: str = Field(..., description="送入 Embedding Engine 的原始重組訊息")
    vector_dim: int = Field(..., description="語義向量的總維度，如 384")
    vector_preview: List[float] = Field(..., description="向量前5維矩陣預覽")

class CVFeatures(BaseModel):
    count: int
    results: Dict[str, int]
    detections: List[Dict[str, Any]]

# ==========================================
# 升級後的路由端點
# ==========================================
@router.post("/extract_log_signature", response_model=VectorTriageResponse, status_code=status.HTTP_200_OK)
async def extract_log_signature(request: Request, payload: LogPayload):
    """
    接收日誌，向地端 Ollama 索取語義特徵圖，
    同時回傳該向量對應的結構化訊息與維度元數據。
    """
    try:
        # ⚡ 從狀態機動態撈出日誌嵌入服務
        agent_service = request.app.state.models["llm_agent"]
        
        # 結構化重組，這就是我們送入下游比對矩陣的「特徵標體」
        log_content = f"Server {payload.server_id}: {payload.error_msg}"
        
        # 點火非同步管道，遇到 await 時讓出 Event Loop 核心，切去隔壁處理 CV 產線搬進來的圖片
        log_vector = await agent_service.get_log_vector(log_content)
        
        # 🤝 架構升級：完美封裝向量與其對應訊息
        return VectorTriageResponse(
            server_id=payload.server_id,
            raw_content=log_content,
            vector_dim=len(log_vector),
            vector_preview=log_vector[:5]  # 預覽前5個維度矩陣
        )
        
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lifespan models pool not properly initialized."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Downstream Embedding Engine Error: {str(e)}"
        )

@router.post("/analyze_objects")
async def analyze_objects(request: Request, features: CVFeatures):
    """接收 YOLOv11 的偵測結果，交給 LLM 進行分析"""
    agent_service = request.app.state.models["llm_agent"]
    yolo_serialized_features = features.model_dump_json()
    llm_analysis_report = await agent_service.analyze_yolo_features(yolo_serialized_features)
    
    return {
        "count": features.count, 
        "results": features.results,
        "ai_insights": llm_analysis_report
    }