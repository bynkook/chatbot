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
        FastAPI 버전과 동일하게 반복 예측(동시 갱신) 방식 적용.
        """
        # f_idx 계산
        fck = float(inputs.get("fck"))
        fy  = float(inputs.get("fy"))
        try:
            f_idx = float(f"{int(round(fck))}{int(round(fy))}") / 1000.0
        except Exception:
            f_idx = float(inputs.get("f_idx"))

        # 초기 payload
        feat_iter: Dict[str, float] = {
            "f_idx": f_idx,
            "width": float(inputs.get("width")),
            "height": float(inputs.get("height")),
            "phi_mn": float(inputs.get("phi_mn")) if inputs.get("phi_mn") is not None else np.nan,
        }

        last_preds: Dict[str, float] = {}
        MAX_ITERS = 5
        for it in range(MAX_ITERS):
            preds_k: Dict[str, float] = {}
            for tgt in self.targets:
                feats = self.features_by_target[tgt]
                row = {f: feat_iter.get(f, np.nan) for f in feats}
                X = pd.DataFrame([row], columns=feats)
                try:
                    preds_k[tgt] = float(self.models[tgt].predict(X)[0])
                except Exception:
                    preds_k[tgt] = np.nan

            # --- debug: per-iteration echo ---
            print(f"[DEBUG] iter {it+1}, preds={preds_k}")

            # 동시 갱신: bd, Sm, rho
            if np.isfinite(preds_k.get("bd", np.nan)):
                feat_iter["bd"] = preds_k["bd"]
            if np.isfinite(preds_k.get("Sm", np.nan)):
                feat_iter["Sm"] = preds_k["Sm"]
            if np.isfinite(preds_k.get("rho", np.nan)):
                feat_iter["rho"] = preds_k["rho"]

            last_preds = preds_k

        print("[DEBUG] predict_all final outputs:", last_preds)
        return last_preds
