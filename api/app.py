from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.detector import Detector
from api.services.log_embedding_agent import LogEmbeddingService
from api.services.analysis_agent_openAI import call_cloud_llm_pipeline
from api import ROOT

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 資源池初始化
    app.state.models = {
        "yolo": Detector(ROOT / "weights" / "Proposed" / "best.pt"),
        "llm_analysis": call_cloud_llm_pipeline,
        "llm_embed": LogEmbeddingService()
    }
    yield
    # 清理
    app.state.models.clear()

app = FastAPI(title="UAV-YOLO & AIOps System", lifespan=lifespan)