# chat_app_V2.py
# TODO : 자연어 입력으로 예측값 출력, scikit-learn, xgboost 모듈 설치 버젼 이슈 해결

import os
import json
import re
import time
import streamlit as st
from typing import List, Dict, Optional
from llama_cpp import Llama
# user functions
from model_bundle import ModelBundle
from predict_parser import parse_predict_message

st.set_page_config(page_title="BKChat Local", layout="wide")
st.markdown("""
<style>
/* 전체 타이포/링크 */
html, body, [class*="block-container"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
a { color:#9E7BFF !important; }

/* 카드/컨테이너 여백 */
.stMainBlockContainer { padding-top: 1rem; }

/* 입력창, select, number input */
div[data-baseweb="input"] > div, div[role="spinbutton"], textarea {
  background:#0F1531 !important; border:1px solid #283059 !important; color:#E6E6F0 !important; border-radius:10px;
}
/* 슬라이더 */
div.stSlider > div[data-baseweb="slider"] > div {
    background-color: transparent !important;
}

/* Slider 진행 부분 색상 변경 (오렌지) */
div.stSlider > div[data-baseweb="slider"] > div > div {
    background-color: #FFA500 !important;
}

/* Slider 핸들 색상 변경 (오렌지) */
div.stSlider [role="slider"] {
    background-color: #FF8C00 !important;
    border: 2px solid #FF8C00 !important;
}

/* 버튼 */
button[kind="primary"] {
  background: linear-gradient(135deg,#7C4DFF 0%, #5B8CFF 100%) !important;
  color:#fff !important; border:0 !important; border-radius:12px !important;
}
button[kind="secondary"] {
  background:#1B2246 !important; color:#E6E6F0 !important; border:1px solid #2A3370 !important; border-radius:12px !important;
}
button:hover { filter:brightness(1.05); }

/* 사이드바 */
section[data-testid="stSidebar"] {
  background:#0C1230 !important; border-right:1px solid #202A55;
}

/* 채팅 버블 */
div[data-testid="stChatMessage"] {
  background:#10173A; border:1px solid #263066; border-radius:14px; padding:12px 14px; box-shadow: 0 6px 20px rgba(0,0,0,.25);
}
div[data-testid="stChatMessage"][data-testid="stChatMessage"] p { color:#E6E6F0; }
/* user/assistant 아이콘 색 대비 */
div[data-testid="stChatMessage"] svg { fill:#9E7BFF !important; }

/* 스크롤바 */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:#2A3370; border-radius:8px; }
::-webkit-scrollbar-track { background:#0B1026; }
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def auto_chat_format(model_path: str, user_choice: str) -> Optional[str]:
    if user_choice == "auto":  # 파일명으로 추정
        name = os.path.basename(model_path).lower()
        if "qwen" in name or "deepseek" in name: return "qwen"
        if "llama-3" in name or "llama3" in name: return "llama-3"
        return None
    if user_choice == "none": return None
    return user_choice

def strip_think(text: str) -> str:
    # 필요 시 체크박스로 On/Off
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

@st.cache_resource(show_spinner="Loading model...")
def load_llm(model_path: str, n_ctx: int, n_threads: int, chat_format: Optional[str]) -> Llama:
    try:
        return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                     chat_format=chat_format, verbose=False)
    except Exception as e:
        if "Invalid chat handler" in str(e):
            return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         chat_format=None, verbose=False)
        raise

def _split_reasoning(text: str) -> tuple[str, str]:
    """모델 응답에서 <think> 블록을 분리한다. visible, think 를 반환."""
    thinks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    think_text = "\n\n".join(t.strip() for t in thinks) if thinks else ""
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return visible, think_text

def chat_once(llm: Llama, messages: List[Dict[str,str]], temperature: float, top_p: float, max_tokens: int, stream_placeholder):
    # 스트리밍 출력
    try:
        it = llm.create_chat_completion(messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stream=True)
        buf=[]
        for ch in it:
            delta = ch["choices"][0]["delta"].get("content","")
            if delta:
                buf.append(delta)
                stream_placeholder.markdown("".join(buf))
        raw = "".join(buf)
        return _split_reasoning(raw)
    except TypeError:
        resp = llm.create_chat_completion(messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        raw = resp["choices"][0]["message"]["content"]
        stream_placeholder.markdown(raw)
        return _split_reasoning(raw)

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model (.gguf) path", r"d:\mycode\chatbot\model\Qwen3-4B-Instruct-2507-Q3_K_S.gguf")
    chat_fmt_choice = st.selectbox("chat_format", ["auto","qwen","llama-3","none"], index=0)
    ctx = st.number_input("n_ctx", 256, 8192, 2048, 256)
    threads = st.number_input("n_threads", 1, 64, max(1,(os.cpu_count() or 4)-1), 1)
    temp = st.slider("temperature", 0.0, 1.5, 0.6, 0.05)
    topp = st.slider("top_p", 0.1, 1.0, 0.95, 0.01)
    toks = st.number_input("max_tokens", 16, 4096, 512, 16)
    # 새 메시지에만 적용되는 기본 접힘 옵션
    hide_think_default = st.checkbox("Hide reasoning by default", value=True, help="변경은 새 응답에만 적용됩니다.")
    reload_btn = st.button("Reload model", use_container_width=True)
    st.divider()
    sys_default = "당신은 한국어와 영어를 명확하고 간결하게 답하는 조수입니다."
    system_prompt = st.text_area("System prompt", sys_default, height=80)
    clear_chat = st.button("Clear chat", use_container_width=True)
    # Load chat(JSON)
    uploaded = st.file_uploader("Load chat (JSON)", type=["json"], key="load_json_sidebar")
    if uploaded is not None:
        try:
            d = json.loads(uploaded.read().decode("utf-8"))
            st.session_state.history = d.get("messages", [])
            st.success("Loaded chat from JSON.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Load failed: {e}")
    st.caption("모델과 설정이 바뀌면 Reload를 누르세요.")

    # --- Regression bundle (minimal UI 추가) ---
    st.subheader("🧮 Regression bundle")
    if "bundle" not in st.session_state:
        st.session_state.bundle = None
    bundle_path = st.text_input("bundle (.joblib) path", r"d:\mycode\rcmodel\output\xgb_bundle.joblib", key="bundle_path")    
    col_b1, col_b2 = st.columns([1,1])
    with col_b1:
        enable_predict = st.checkbox("Enable /predict", value=True, key="enable_predict")
    with col_b2:
        load_bundle_btn = st.button("Load bundle", width="stretch", key="btn_load_bundle")

    if load_bundle_btn:
        try:
            st.session_state.bundle = ModelBundle.load(bundle_path)
            st.success(f"Bundle loaded: {', '.join(st.session_state.bundle.targets)}")
        except Exception as e:
            st.session_state.bundle = None
            st.error(f"Bundle load failed: {e}")

# ---------- state ----------
if "history" not in st.session_state:
    st.session_state.history = []
if clear_chat:
    st.session_state.history = []
if reload_btn:
    st.cache_resource.clear()

# ---------- model ----------
if not os.path.exists(model_path):
    st.error(f"Model not found: {model_path}")
    st.stop()

chat_format = auto_chat_format(model_path, chat_fmt_choice)
llm = load_llm(model_path, ctx, int(threads), chat_format)

# ---------- UI ----------
st.title("BKChat Local")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        # 저장 당시 expanded 상태로만 표시(과거 메시지는 토글 변경의 영향을 받지 않음)
        if turn.get("role") == "assistant" and turn.get("think"):
            with st.expander("Show reasoning", expanded=bool(turn.get("expanded", False))):
                st.markdown(turn["think"])

user_msg = st.chat_input("메시지를 입력하세요…")
if user_msg:
    # 사용자 메시지 출력
    st.session_state.history.append({"role":"user","content":user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # /predict 분기: 회귀 예측 우선 처리
    did_predict = False

    # --- 진단 가드: /predict인데 비활성/미로드면 바로 경고 ---
    if user_msg.strip().lower().startswith("/predict") and not (
        st.session_state.get("enable_predict") and st.session_state.get("bundle") is not None
    ):
        with st.chat_message("assistant"):
            if not st.session_state.get("enable_predict"):
                st.warning("Enable /predict 를 체크하세요.")
            if st.session_state.get("bundle") is None:
                st.warning("회귀 번들이 로드되지 않았습니다. 사이드바에서 joblib 경로를 지정하고 **Load bundle** 을 누르세요.")
        did_predict = True

    if not did_predict and st.session_state.get("enable_predict") and st.session_state.get("bundle") is not None:
        parsed = parse_predict_message(user_msg or "")
        if parsed is not None:  # 실제 /predict 명령으로 인식됨
            with st.chat_message("assistant"):
                try:
                    # 디버그: 파싱된 입력 요약 표시
                    st.caption(f"parsed: { {k: parsed.get(k) for k in ['fck','fy','width','height','phi_mn','Sm','bd','rho']} }")

                    preds = st.session_state.bundle.predict_all(parsed)
                    import pandas as pd
                    df_show = pd.DataFrame([preds], columns=["Sm","bd","rho","phi_mn"])
                    st.markdown("**예측 결과**")
                    st.dataframe(df_show, width="stretch")
                    st.session_state.history.append({
                        "role":"assistant",
                        "content": "예측 결과: " + ", ".join(f"{k}={v:.6g}" for k, v in preds.items())
                    })
                    did_predict = True
                except Exception as e:
                    st.error(f"예측 실패: {e}")
                    st.session_state.history.append({
                        "role":"assistant",
                        "content": f"예측 실패: {e}"
                    })
                    did_predict = True
        else:
            # /predict 이지만 파서가 인식 못한 경우
            with st.chat_message("assistant"):
                st.warning("`/predict` 형식을 인식하지 못했습니다. 예시: `/predict fck=27 fy=400 width=300 height=500 phi_mn=120`")
                st.caption(f"raw: {user_msg!r}")
            did_predict = True

    # 모델 응답 (/predict 가 아니거나 비활성인 경우만)
    if not did_predict:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            msgs = [{"role":"system","content":system_prompt}, *st.session_state.history]
            visible, think = chat_once(llm, msgs, float(temp), float(topp), int(toks), placeholder)
            expanded_now = not hide_think_default
            placeholder.markdown(visible)
            if think:
                with st.expander("Show reasoning", expanded=expanded_now):
                    st.markdown(think)
            st.session_state.history.append({
                "role":"assistant",
                "content": visible,
                "think": think,
                "expanded": expanded_now
            })

# 다운/업로드(대화 저장/불러오기)
if st.button("Save chat"):
    data = {"system": system_prompt, "messages": st.session_state.history}
    st.download_button("Download JSON",
                       data=json.dumps(data, ensure_ascii=False, indent=2),
                       file_name=f"chat_{int(time.time())}.json",
                       mime="application/json",
                       use_container_width=True)
