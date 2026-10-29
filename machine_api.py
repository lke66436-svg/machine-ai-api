
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Smart Conveyor Fault Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://codepen.io",
        "https://cdpn.io",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

# 1. 定義機台輸入資料
class MachineData(BaseModel):
    machine_id: str = "M001"
    current: float = Field(ge=0)
    vibration: float = Field(ge=0)
    rpm: float = Field(ge=0)
    running: bool

    # 新增輸送帶感測資料
    belt_offset: Optional[float] = None
    belt_health: Optional[float] = Field(
        default=None, ge=0, le=100
    )
    roller_stuck: Optional[bool] = None

    # 模擬器故障停機後保留的故障快照
    fault_snapshot: bool = False

# 2. 故障代碼與分析
ERROR_RULES = {
    "E001": {
        "component": "motor",
        "message": "馬達過載",
        "analysis": "可能與負載過重或傳動阻力增加有關"
    },
    "E002": {
        "component": "motor",
        "message": "異常震動",
        "analysis": "可能與軸承磨損或零件鬆動有關"
    },
    "E003": {
        "component": "motor",
        "message": "轉速異常",
        "analysis": "可能與負載或轉速控制問題有關"
    },
    "E004": {
        "component": "belt",
        "message": "皮帶偏移",
        "analysis": "可能與皮帶張力或滾輪不對中有關"
    },
    "E005": {
        "component": "belt",
        "message": "皮帶損耗",
        "analysis": "可能與皮帶老化或長期磨耗有關"
    },
    "E006": {
        "component": "roller",
        "message": "滾輪卡滯",
        "analysis": "可能有異物阻塞或滾輪軸承異常"
    }
}

@app.get("/health")
def health():
    return {"status": "ok"}

# 3. 機台故障偵測
@app.post("/diagnose")
def diagnose(data: MachineData):

    codes = []

    # 運轉中，或模擬器保留的故障快照
    should_check = data.running or data.fault_snapshot

    if should_check:

        if data.current > 4.2:
            codes.append("E001")

        if data.vibration > 3.5:
            codes.append("E002")

        if data.rpm < 1400 or data.rpm > 1500:
            codes.append("E003")

        if (
            data.belt_offset is not None
            and abs(data.belt_offset) > 7
        ):
            codes.append("E004")

        if (
            data.belt_health is not None
            and data.belt_health < 60
        ):
            codes.append("E005")

        if data.roller_stuck is True:
            codes.append("E006")

    errors = [
        {"code": code, **ERROR_RULES[code]}
        for code in codes
    ]

    analysis = (
        "；".join(e["analysis"] for e in errors)
        if errors
        else "未偵測到超出目前設定閾值的異常"
    )

    return {
        "machine_id": data.machine_id,
        "status": "ERROR" if errors else "NORMAL",
        "error_count": len(errors),
        "errors": errors,
        "ai_analysis": analysis,
        "analysis_mode": "SIMULATION"
    }
