# model_bundle.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class ModelBundle:
    models: Dict[str, Any]
    features_by_target: Dict[str, list]
    #dtypes_by_target: Dict[str, Dict[str, str]]
    targets: list

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        b = joblib.load(path)
        # 방어적 체크
        for k in ["models", "features_by_target", "targets"]:   # bundle 구조
            if k not in b:
                raise ValueError(f"bundle missing key: {k}")
        return cls(
            models=b["models"],
            features_by_target=b["features_by_target"],
            #dtypes_by_target=b["dtypes_by_target"],
            targets=b["targets"],
        )

    def predict_all(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        inputs: keys 포함 권장 -> fck, fy, width, height, phi_mn
        내부에서 f_idx 산출 후 각 타깃별 필요 피처만 추출하여 예측.
        """
        # 필수 변환: f_idx = concat(fck, fy)/1000  (예: 27, 400 -> 27400/1000 = 27.4)
        fck = float(inputs.get("fck"))
        fy  = float(inputs.get("fy"))
        try:
            f_idx = float(f"{int(round(fck))}{int(round(fy))}") / 1000.0
        except Exception:
            # fallback: 단순 조합 불가 시 사용자가 직접 f_idx를 준 경우 우선
            f_idx = float(inputs.get("f_idx"))
        # 기본 피처 풀
        base = {
            "f_idx": f_idx,
            "width": float(inputs.get("width")),
            "height": float(inputs.get("height")),
            "Sm": float(inputs.get("Sm")) if inputs.get("Sm") is not None else np.nan,
            "bd": float(inputs.get("bd")) if inputs.get("bd") is not None else np.nan,
            "rho": float(inputs.get("rho")) if inputs.get("rho") is not None else np.nan,
            "phi_mn": float(inputs.get("phi_mn")) if inputs.get("phi_mn") is not None else np.nan,
        }

        out: Dict[str, float] = {}
        for tgt in self.targets:
            feats = self.features_by_target[tgt]
            row = {k: base.get(k, np.nan) for k in feats}
            X = pd.DataFrame([row], columns=feats)

            # 파이프라인 전처리가 결측을 처리하므로 결측 허용
            yhat = float(self.models[tgt].predict(X)[0])
            out[tgt] = yhat
        return out
