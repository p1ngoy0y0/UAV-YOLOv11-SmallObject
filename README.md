# Optimization of Small Object Detection in UAV Aerial Imagery

## 🚀 專案說明 (Project Overview)
本專案旨在將研究室模型，以簡化方式實際部屬為可應用之 API 服務，實現無人機空拍影像偵測的端到端（E2E）推理管線。

## 🔬 論文簡述
針對 VisDrone 空拍場景，為解決傳統模型在微小目標上的特徵流失問題，採用結合 **BiFPN 加權特徵融合機制** 與 **SPD-Conv 無損下採樣模組** 之改良版 YOLOv11。本模型能更有效地保留微小目標的特徵資訊，同時維持輕量化優勢。
*   **mAP@50**: 達到 **50.40%**，相較於原版 YOLOv11-s 提升了 **10.09%**。
*   **模型尺寸**: 參數量僅微幅增加至 **12.02M**，證明了設計的高效性。

### 研究核心改進 (Key Innovations)

*   **BiFPN (Weighted Feature Fusion)**: 採用雙向加權特徵金字塔結構，能更有效地融合深層語義與淺層細節特徵，且特徵融合過程更具效率。
*   **SPD-Conv (Lossless Downsampling)**: 藉由 **Space-to-Depth** 機制取代會造成資訊流失的步長卷積，使模型在連續下採樣後仍能保留小物件的細微邊界與紋理資訊，解決小目標「特徵消失」的痛點。

### 研究實驗結果 (Performance Comparison)

本研究在 **VisDrone-DET2019** 驗證集上進行測試，結果顯示結合架構改良與推理策略後，模型效能獲得顯著提升。

| 模型架構組合 | mAP@50 (%) | 提升幅度 (Improvement) | 參數量 (Params) |
| :--- | :---: | :---: | :---: |
| YOLOv11s (Baseline) | 40.31 | - | 9.43M |
| YOLOv11s + BiFPN | 47.07 | +6.76 | 8.06M |
| YOLOv11s + SPD-Conv | 42.12 | +1.81 | 10.62M |
| **UAV-YOLOv11 (Proposed)** | **50.40** | **+10.09** | **12.02M** |

**Note**：
1. **架構改良**：引入 BiFPN 與 SPD-Conv 後，純模型效能從 40.31% 提升至 50.40%。
2. **輕量化**：在大幅提升精度的同時，參數量僅微幅增加至 12.02M，極具邊緣端部署潛力。

## ⚙️延伸專案說明
本部分實現將研究模型落地為高並發、高可用性的邊緣推論服務，服務端採用 FastAPI 異步架構，實現視覺推論與 LLM 診斷引擎的解耦與非同步調度。

### 核心技術

*   **Asymmetric Compute Pipeline**:
    *   I/O Bound: 運用 async/await 處理高並發 API 請求。
    *   Compute Bound: 將視覺推論與 LLM 推論分離，避免 Event Loop 阻塞。
*   **Containerized Orchestration**: 透過 Docker Multi-stage Build 與 docker-compose 定義微服務環境，確保研究室模型在不同生產環境的可重現性。
*   **SoC Architecture**: 嚴格執行「關注點分離 (Separation of Concerns)」，將 AI 引擎（算力密集）與 API 服務（請求分發）解耦，利於未來真正佈署於邊緣運算裝置（如 NVIDIA Jetson 或工業級嵌入式運算單元）。

## 資料夾結構 (Repository Structure)：
```bash
├── api/                            #FastAPI 服務
│   ├──routers                      
│   │   ├──agent_router.py          #LLM 路由
│   │   └──cv_router.py             #視覺推論路由
│   ├──services
│   │   └──agent_service.py         #LLM agent
│   ├──__init__.py
│   ├── app.py
│   ├── detector.py
│   └── main.py
├── data/                           # 資料集路徑
├── models/                         # 模型架構定義 (YAML)
│   ├── yolo11_BiFPN_s.yaml
│   ├── yolo11_BiFPN_SPDConv.yaml
│   └── yolo11_SPD.yaml
├── ultralytics/                    # 核心運算引擎包
├── weights/                        # 訓練權重
│   ├── BiFPN/
│   ├── Proposed/
│   ├── SPDConv/
│   └── YOLOv11/
├── .gitignore                      # 版本控制忽略清單
├── docker-compose.yml              # Docker Compose 設定檔
├── dockerfile                      # Dockerfile (FastAPI)
├── dockerfile.ollama               # Dockerfile (LLM)
├── README.md                       # 專案說明文件
├── requirements.txt                # 環境依賴清單
└── Thesis.pdf                      # 論文內容簡介(A0海報格式)
```