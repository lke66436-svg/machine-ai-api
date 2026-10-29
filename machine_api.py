from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Machine Error Detection API")

# 允許 Three.js 連接
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

# 機台資料
class MachineData(BaseModel):
    machine_id: str
    current: float = Field(ge=0)
    vibration: float = Field(ge=0)
    rpm: float = Field(ge=0)
    running: bool

# 故障代碼
ERROR_RULES = {
    "E001": {
        "component": "motor",
        "message": "電流過高",
        "analysis": "可能與機械負載過重有關"
    },
    "E002": {
        "component": "motor",
        "message": "震動異常",
        "analysis": "可能與軸承或傳動系統有關"
    },
    "E003": {
        "component": "motor",
        "message": "轉速異常",
        "analysis": "可能與轉速控制異常有關"
    }
}

@app.get("/health")
def health():
    return {"status": "ok"}

# 機台故障偵測 API
@app.post("/diagnose")
def diagnose(data: MachineData):
    codes = []

    if data.running:
        if data.current > 15:
            codes.append("E001")

        if data.vibration > 2.0:
            codes.append("E002")

        if not 1400 <= data.rpm <= 1600:
            codes.append("E003")

    errors = [
        {"code": code, **ERROR_RULES[code]}
        for code in codes
    ]

    analysis = (
        "；".join(e["analysis"] for e in errors)
        if errors
        else "未偵測到超出設定閾值的異常"
    )

    return {
        "machine_id": data.machine_id,
        "status": "ERROR" if errors else "NORMAL",
        "error_count": len(errors),
        "errors": errors,
        "ai_analysis": analysis,
        "analysis_mode": "SIMULATION"
    }
