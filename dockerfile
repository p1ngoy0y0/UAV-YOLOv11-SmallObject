FROM python:3.8

WORKDIR /workspace

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# 複製環境清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製 Pack 與權重
COPY api/ ./api/
COPY ultralytics/ ./ultralytics/
COPY weights/Proposed/ ./weights/Proposed/

# FastAPI 預設的 8000 埠口
EXPOSE 8000

# 啟動命令
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]