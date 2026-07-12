from api.app import app
from api.routers.cv_router import router as cv_router
from api.routers.agent_router import router as agent_router

app.include_router(cv_router)
app.include_router(agent_router)