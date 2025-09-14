# predict_parser.py
import re
from typing import Dict, Optional

KEYMAP = {
    "fck": ["fck", "콘크리트"],
    "fy": ["fy", "철근"],
    "width": ["width", "b", "bw", "단면폭", "폭"],
    "height": ["height", "h", "단면높이", "높이"],
    "phi_mn": ["phi_mn", "mu", "휨모멘트", "모멘트"],
    "Sm": ["Sm"],
    "bd": ["bd"],
    "rho": ["rho"],
}

def parse_predict_message(text: str) -> Optional[Dict[str, float]]:
    """
    허용 형식:
      /predict fck=27 fy=400 width=300 height=500 phi_mn=120
      /predict {"fck":27, "fy":400, "width":300, "height":500, "phi_mn":120}
      /predict fck:27, fy:400, b=300, h=500, mu=120
    """
    if not text.strip().lower().startswith("/predict"):
        return None
    payload = text.strip()[len("/predict"):].strip()
    if not payload:
        return {}

    # JSON 스타일 시도
    if payload[0] in "{[":
        try:
            import json
            d = json.loads(payload)
            return _normalize_keys(d)
        except Exception:
            pass

    # key=value or key:value 토큰 파싱
    tokens = re.findall(r'([\w\-\u3131-\u318E\uAC00-\uD7A3]+)\s*[:=]\s*([-+]?[\d\.]+)', payload)
    d = {k: float(v) for k, v in tokens}
    return _normalize_keys(d)

def _normalize_keys(d: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for std, aliases in KEYMAP.items():
        for a in aliases:
            for k in d.keys():
                if k.lower() == a.lower():
                    out[std] = float(d[k])
                    break
        # 누락 가능: Sm, bd, rho는 입력 없이 예측할 수 있도록 NaN 허용
    return out
