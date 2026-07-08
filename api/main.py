from api.app import app
from api.routers.cv_router import router as cv_router
from api.routers.triage import router as triage_router

app.include_router(cv_router)
app.include_router(triage_router)