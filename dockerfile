# 1. 指定維護中的 Python 版本 (Bullseye)，避免 EOL 帶來的連結錯誤
FROM python:3.8-slim-bullseye

WORKDIR /workspace

# 2. 設定非互動環境變數，防止安裝時發生 interactive prompt 卡死
ENV DEBIAN_FRONTEND=noninteractive

# 3. 升級 apt 軟體源，安裝必要依賴，並清理緩存以減小體積
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zstd \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. 複製環境與業務邏輯
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-multipart

COPY api/ ./api/
COPY ultralytics/ ./ultralytics/
COPY weights/Proposed/ ./weights/Proposed/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]