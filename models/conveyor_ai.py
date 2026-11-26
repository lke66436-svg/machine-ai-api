from pathlib import Path
import math

import joblib
import pandas as pd


# ==========================================
# 1. V2 模型與特徵設定
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


# ==========================================
# 2. 六種故障分類
# ==========================================

FAULT_NAMES = {
    "Normal": "正常運作",
    "Motor_Fault": "馬達過載",
    "Belt_Misalignment": "皮帶偏移",
    "Belt_Wear": "皮帶磨損",
    "Conveyor_Jam": "滾輪卡滯",
    "Abnormal_Vibration": "異常震動",
}

EXPECTED_CLASSES = set(FAULT_NAMES.keys())


# ==========================================
# 3. 3D 模擬器故障模式對照
# ==========================================

FAULT_MODE_MAP = {
    "normal": "Normal",
    "overload": "Motor_Fault",
    "shift": "Belt_Misalignment",
    "wear": "Belt_Wear",
    "roller": "Conveyor_Jam",
    "vibration": "Abnormal_Vibration",
}


# ==========================================
# 4. 感測器資料範圍
# ==========================================

# 以下數值依照 3D 模擬器設定。
# 不代表真實工業設備的標準規格。

SENSOR_RANGES = {

    "normal": {
        "Motor_Current": (2.2, 3.5),
        "Vibration": (0.6, 1.8),
        "RPM": (1430, 1470),
        "Belt_Offset": (-1.5, 1.5),
        "Belt_Health": (100, 100),
    },

    "overload": {
        "Motor_Current": (4.5, 6.5),
        "Vibration": (2.0, 4.0),
        "RPM": (1250, 1400),
        "Belt_Offset": (-3, 3),
        "Belt_Health": (100, 100),
    },

    "shift": {
        "Motor_Current": (2.5, 3.8),
        "Vibration": (1.5, 3.5),
        "RPM": (1400, 1460),
        "Belt_Offset": [(-18, -8), (8, 18)],
        "Belt_Health": (95, 95),
    },

    "wear": {
        "Motor_Current": (2.5, 3.8),
        "Vibration": (1.5, 3.5),
        "RPM": (1400, 1460),
        "Belt_Offset": (-8, 8),
        "Belt_Health": (35, 35),
    },

    "roller": {
        "Motor_Current": (4.5, 6.5),
        "Vibration": (4.0, 8.0),
        "RPM": (0, 500),
        "Belt_Offset": (-8, 8),
        "Belt_Health": (100, 100),
    },

    "vibration": {
        "Motor_Current": (2.5, 3.8),
        "Vibration": (4.0, 8.0),
        "RPM": (1400, 1470),
        "Belt_Offset": (-4, 4),
        "Belt_Health": (98, 100),
    },
}


# ==========================================
# 5. 載入 V2 AI 模型
# ==========================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到 V2 AI 模型：{MODEL_PATH}\n"
            "請先執行 train_model_v2.py 完成模型訓練。"
        )

    loaded_model = joblib.load(MODEL_PATH)

    if not callable(getattr(loaded_model, "predict", None)):
        raise TypeError("模型缺少 predict() 方法。")

    if not callable(getattr(loaded_model, "predict_proba", None)):
        raise TypeError("模型缺少 predict_proba() 方法。")

    # 確認輸入特徵
    model_features = getattr(
        loaded_model,
        "feature_names_in_",
        None
    )

    if model_features is not None:

        if list(model_features) != FEATURES:
            raise ValueError(
                "模型輸入特徵不一致。\n"
                f"模型特徵：{list(model_features)}\n"
                f"目前特徵：{FEATURES}"
            )

    # 確認六種分類標籤
    model_classes = set(
        str(label) for label in loaded_model.classes_
    )

    if model_classes != EXPECTED_CLASSES:
        raise ValueError(
            "模型故障分類不一致。\n"
            f"模型分類：{sorted(model_classes)}\n"
            f"預期分類：{sorted(EXPECTED_CLASSES)}"
        )

    return loaded_model


model = load_model()


# ==========================================
# 6. 感測器資料驗證
# ==========================================

def validate_sensor_data(
    motor_current,
    vibration,
    rpm,
    belt_offset,
    belt_health,
):

    sensor_values = {
        "Motor_Current": motor_current,
        "Vibration": vibration,
        "RPM": rpm,
        "Belt_Offset": belt_offset,
        "Belt_Health": belt_health,
    }

    validated_values = {}

    for name, value in sensor_values.items():

        if isinstance(value, bool):
            raise ValueError(
                f"{name} 必須是有效數字。"
            )

        try:
            number = float(value)

        except (TypeError, ValueError):
            raise ValueError(
                f"{name} 必須是有效數字。"
            ) from None

        if not math.isfinite(number):
            raise ValueError(
                f"{name} 必須是有限數字。"
            )

        # 皮帶偏移允許正負數
        if name != "Belt_Offset" and number < 0:
            raise ValueError(
                f"{name} 不可為負數。"
            )

        # 皮帶健康度限制 0～100
        if name == "Belt_Health":

            if not 0 <= number <= 100:
                raise ValueError(
                    "Belt_Health 必須介於 0～100。"
                )

        validated_values[name] = number

    return validated_values


# ==========================================
# 7. AI 診斷核心
# ==========================================

def diagnose(
    motor_current,
    vibration,
    rpm,
    belt_offset,
    belt_health,
):

    sensor_data = validate_sensor_data(
        motor_current,
        vibration,
        rpm,
        belt_offset,
        belt_health,
    )

    input_data = pd.DataFrame(
        [[sensor_data[name] for name in FEATURES]],
        columns=FEATURES,
    )

    prediction = str(
        model.predict(input_data)[0]
    )

    probabilities = model.predict_proba(input_data)[0]

    scores = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities
        )
    }

    return {
        "fault_type": prediction,
        "fault_name": FAULT_NAMES[prediction],
        "scores": scores,
        "sensor_data": sensor_data,
        "note": (
            "本系統使用依照 3D 模擬器感測器範圍"
            "建立的模擬資料訓練 AI 模型，"
            "診斷結果僅供專題展示與設備狀態分析。"
        ),
    }


# ==========================================
# 8. 接收 3D 模擬器感測器資料
# ==========================================

def diagnose_3d_data(data):

    """
    支援 3D 模擬器及 API 使用的感測器欄位。

    支援格式：

    current
    vibration
    rpm
    beltOffset
    beltHealth

    或：

    current
    vibration
    rpm
    belt_offset
    belt_health
    """

    # 同時支援 camelCase 與 snake_case
    field_mapping = {
        "current": ["current"],
        "vibration": ["vibration"],
        "rpm": ["rpm"],
        "beltOffset": ["beltOffset", "belt_offset"],
        "beltHealth": ["beltHealth", "belt_health"],
    }

    sensor_values = {}

    for target_field, possible_fields in field_mapping.items():

        found = False

        for field in possible_fields:

            if field in data:
                sensor_values[target_field] = data[field]
                found = True
                break

        if not found:
            raise ValueError(
                f"缺少 3D 感測器資料：{possible_fields}"
            )

    return diagnose(
        motor_current=sensor_values["current"],
        vibration=sensor_values["vibration"],
        rpm=sensor_values["rpm"],
        belt_offset=sensor_values["beltOffset"],
        belt_health=sensor_values["beltHealth"],
    )


# ==========================================
# 9. 故障模式轉換
# ==========================================

def get_fault_class(fault_mode):

    if fault_mode not in FAULT_MODE_MAP:
        raise ValueError(
            f"未知的 3D 故障模式：{fault_mode}"
        )

    return FAULT_MODE_MAP[fault_mode]


# ==========================================
# 10. 單獨執行測試
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("智慧輸送帶 AI 診斷系統 V2")
    print("=" * 55)

    test_data = {
        "current": 5.5,
        "vibration": 3.0,
        "rpm": 1320,
        "belt_offset": 0,
        "belt_health": 100,
    }

    result = diagnose_3d_data(test_data)

    print("\n感測器資料：")

    for name, value in result["sensor_data"].items():
        print(f"{name}: {value}")

    print("\nAI 診斷結果：")

    print(f"故障類別：{result['fault_type']}")
    print(f"診斷名稱：{result['fault_name']}")

    print("\n各類別預測分數：")

    for fault, score in result["scores"].items():
        print(
            f"{FAULT_NAMES[fault]}：{score:.2%}"
        )

    print("\n注意：", result["note"])
