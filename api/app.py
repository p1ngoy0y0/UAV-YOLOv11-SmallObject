from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.detector import Detector
from api.services.agent_service import LocalAgentService
from api import ROOT

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 資源池初始化
    app.state.models = {
        "yolo": Detector(ROOT / "weights" / "Proposed" / "best.pt"),
        "llm_agent": LocalAgentService()
    }
    yield
    # 清理
    app.state.models.clear()

app = FastAPI(title="UAV-YOLO & AIOps System", lifespan=lifespan)