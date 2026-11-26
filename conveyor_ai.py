from pathlib import Path
import math

import joblib
import numpy as np


# ==========================================
# 1. 專案設定
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "conveyor_fault_model_v2.joblib"

FEATURES = [
    "Motor_Current",
    "Vibration",
    "RPM",
    "Belt_Offset",
    "Belt_Health",
]

FAULT_NAMES = {
    "Normal": "正常運作",
    "Motor_Fault": "馬達過載",
    "Belt_Misalignment": "皮帶偏移",
    "Belt_Wear": "皮帶磨損",
    "Conveyor_Jam": "滾輪卡滯",
    "Abnormal_Vibration": "異常震動",
}

FAULT_NOTES = {
    "Normal": "目前感測器數值符合正常運作特徵。",
    "Motor_Fault": "模型判斷可能存在馬達過載，建議檢查馬達電流、負載及轉速。",
    "Belt_Misalignment": "模型判斷可能存在皮帶偏移，建議檢查皮帶位置及導輪。",
    "Belt_Wear": "模型判斷可能存在皮帶磨損，建議檢查皮帶表面及健康度。",
    "Conveyor_Jam": "模型判斷可能存在滾輪卡滯，建議檢查滾輪、傳動機構及輸送帶是否受阻。",
    "Abnormal_Vibration": "模型判斷可能存在異常震動，建議檢查馬達、軸承及機構固定狀態。",
}

# 模型只載入一次，避免每次診斷都重新讀取檔案。
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"找不到 AI 模型檔案：{MODEL_PATH}\n"
        "請確認 conveyor_fault_model_v2.joblib 位於 models 資料夾。"
    )

_model = joblib.load(MODEL_PATH)

# 確認模型具有本程式預期的特徵順序及六種類別。
if hasattr(_model, "n_features_in_") and _model.n_features_in_ != len(FEATURES):
    raise ValueError(
        f"模型特徵數量不符：預期 {len(FEATURES)}，"
        f"實際 {_model.n_features_in_}。"
    )

_model_classes = set(getattr(_model, "classes_", []))
_expected_classes = set(FAULT_NAMES)
if _model_classes and _model_classes != _expected_classes:
    raise ValueError(
        "模型分類標籤與 V2 設定不一致。"
        f"模型標籤：{sorted(_model_classes)}"
    )


# ==========================================
# 2. 輸入資料驗證
# ==========================================

def _validate_sensor_data(
    motor_current,
    vibration,
    rpm,
    belt_offset,
    belt_health,
):
    """驗證五項感測器數值，並回傳標準化後的 float。"""

    values = {
        "motor_current": motor_current,
        "vibration": vibration,
        "rpm": rpm,
        "belt_offset": belt_offset,
        "belt_health": belt_health,
    }

    converted = {}

    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} 必須是數值。")

        if not math.isfinite(number):
            raise ValueError(f"{name} 必須是有限數值，不可為 NaN 或 Infinity。")
        