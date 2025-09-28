# predict_parser.py
import re
from typing import Dict, Optional

KEYMAP = {
    "fck": ["fck", "콘크리트", "콘크리트강도", "콘크리트압축강도", "콘크리트 강도", "콘크리트 압축강도", "콘크리트 설계강도"],
    "fy": ["fy", "철근", "철근강도", "철근 강도", "철근 규격", "철근 항복강도", "철근항복강도", "rebar class"],
    "width": ["width", "b", "bw", "단면폭", "폭"],
    "height": ["height", "h", "단면높이", "높이"],
    "phi_mn": ["phi_mn", "mu", "휨모멘트", "모멘트"],
    "Sm": ["Sm", "단면계수"],
    "bd": ["bd", "단면폭깊이"],
    "rho": ["rho", "철근비"],
}

# 예측에 필수로 필요한 기본 입력 정의
REQUIRED_INPUT = ['fck', 'fy', 'width', 'height', 'phi_mn']

def parse_predict_message(text: str) -> Optional[Dict[str, float]]:
    """
    허용 형식: (단위 : mm, MPa, kN-m)
      /predict fck=27 fy=400 width=800 height=1000 phi_mn=1000
      /predict {"fck":27, "fy":400, "width":800, "height":1000, "phi_mn":1000}
      /predict fck:27, fy:400, b=800, h=1000, mu=1000
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


# -------------------------------
# 자연어 파서 추가
# -------------------------------
_NUM = r"([-+]?\d+(?:\.\d+)?)"
_SP = r"[ \t]*"
_UNIT_MPA = r"(?:MPa|메가파스칼)?"
_UNIT_MM = r"(?:mm|밀리미터)?"
_UNIT_KNM = r"(?:kN[\-\s·]?m|kNm|킬로뉴턴미터)?"

_NUM = r"([-+]?\d+(?:\.\d+)?)"
_SP = r"[ \t]*"
_UNIT_MPA = r"(?:MPa|메가파스칼)?"
_UNIT_MM = r"(?:mm|밀리미터)?"
_UNIT_KNM = r"(?:kN[\-\s·]?m|kNm|킬로뉴턴미터)?"

_PATTERNS = {
    # 단어 경계 적용: \b 또는 lookaround로 약어(b,h,mu 등)가 단어 내부(width, bd 등)에 매칭되지 않도록 함
    "fck": re.compile(rf"(?:콘크리트\s*강도|설계\s*압축강도|\bfck\b){_SP}[:=]?\s*{_NUM}\s*{_UNIT_MPA}", re.IGNORECASE),
    "fy": re.compile(rf"(?:철근\s*항복강도|항복\s*강도|\bfy\b){_SP}[:=]?\s*{_NUM}", re.IGNORECASE),
    "width": re.compile(rf"(?:단면\s*폭|폭|\bwidth\b|\bbw\b|(?<![A-Za-z])b(?![A-Za-z])){_SP}[:=]?\s*{_NUM}\s*{_UNIT_MM}", re.IGNORECASE),
    "height": re.compile(rf"(?:단면\s*높이|높이|\bheight\b|(?<![A-Za-z])h(?![A-Za-z])){_SP}[:=]?\s*{_NUM}\s*{_UNIT_MM}", re.IGNORECASE),
    "phi_mn": re.compile(rf"(?:휨\s*모멘트|공칭휨강도|\bmu\b|\bphi[_\-]?mn\b){_SP}[:=]?\s*{_NUM}\s*{_UNIT_KNM}", re.IGNORECASE),
}

def parse_predict_natural(text: str) -> Dict[str, float]:
    """
    한국어/혼합 자연어에서 필수 입력값을 추출한다.
    예: '콘크리트 강도 27 MPa, 철근항복강도 400, 단면 폭 800mm, 높이 1000, 휨모멘트 1000 kN-m'
    """
    d: Dict[str, float] = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(text)
        if m:
            try:
                d[key] = float(m.group(1))
            except Exception:
                pass
    # 선택항목도 키워드 그대로 허용
    # Sm, bd, rho가 자연어에 직접 나오면 'Sm=..' 같은 형식으로 parse_predict_message가 처리
    return d

def is_predict_intent(text: str) -> bool:
    """
    예측의도 판단:
        - '/predict' 로 시작하면 True
        - 아래 키워드 중 2개 이상 포함시 True
    """
    t = text.strip().lower()
    if t.startswith("/predict"):
        return True
    kws = [
        'fck', 'fy', 'phi_mn', 'mu',
        '콘크리트', '강도', '철근', '항복강도', '철근량',
        '단면', '폭', '높이', '휨모멘트', '휨강도', '공칭휨강도',
        'Sm', 'bd', 'rho'
        ]
    hits = sum(1 for k in kws if k in t)
    return hits >= 2


def build_missing_prompt(missing_keys) -> str:
    ex = "/predict fck=27 fy=400 width=800 height=1000 mu=1000"
    labels = {
        "fck": "콘크리트 강도 fck (MPa)",
        "fy": "철근 항복강도 fy (MPa)",
        "width": "단면 폭 width (mm)",
        "height": "단면 높이 height (mm)",
        "phi_mn": "휨모멘트 Mu=phi_mn (kN·m)",
    }
    need = ", ".join(labels[k] for k in missing_keys)
    return (
        f"예측을 위해 다음 항목이 더 필요합니다: {need}\n"
        f"자연어로 입력하거나 예시처럼 입력하세요: {ex}"
    )

