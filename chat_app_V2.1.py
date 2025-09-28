# chat_app_V2.1.py

import os
import json
import re
import time
# import pandas as pd   # only for table expression printing
import streamlit as st
from typing import List, Dict, Optional
from llama_cpp import Llama
# user functions
from model_bundle import ModelBundle
from predict_parser import (
    parse_predict_message,
    parse_predict_natural,
    build_missing_prompt,
    REQUIRED_INPUT,
    is_predict_intent,
)

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

@st.cache_data(show_spinner=False, max_entries = 4)
def load_bundle_cached(path: str) -> ModelBundle:
    '''joblib 번들을 디스크에 로드하고 캐시'''
    return ModelBundle.load(path)

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
    model_path = st.text_input("Model (.gguf) path", r"c:\Users\BKHOME\mycode\chatbot\models\Qwen3-4B-Instruct-2507-Q3_K_S.gguf")
    chat_fmt_choice = st.selectbox("chat_format", ["auto","qwen","llama-3","none"], index=0)
    ctx = st.number_input("n_ctx", 256, 8192, 2048, 256)
    threads = st.number_input("n_threads", 1, 64, max(1,(os.cpu_count() or 4)-1), 1)
    temp = st.slider("temperature", 0.0, 1.5, 0.6, 0.05)
    topp = st.slider("top_p", 0.1, 1.0, 0.95, 0.01)
    toks = st.number_input("max_tokens", 16, 4096, 512, 16)
    # 새 메시지에만 적용되는 기본 접힘 옵션
    hide_think_default = st.checkbox("Hide reasoning by default", value=True, help="변경은 새 응답에만 적용됩니다.")
    reload_btn = st.button("Reload model", width="stretch")
    st.divider()
    sys_default = "당신은 한국어와 영어를 명확하고 간결하게 답하는 조수입니다."
    system_prompt = st.text_area("System prompt", sys_default, height=80)
    clear_chat = st.button("Clear chat", width="stretch")
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
    bundle_path = st.text_input("bundle (.joblib) path", r"c:\Users\BKHOME\mycode\rcmodel\output\stack_bundle_1.joblib", key="bundle_path")
    load_bundle_btn = st.button("Load bundle", width="stretch", key="btn_load_bundle")

    if load_bundle_btn:
        try:
            st.session_state.bundle = load_bundle_cached(bundle_path)
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

    #### 예측 마법사 상태 초기화
    if "predict_wizard" not in st.session_state:
        st.session_state.predict_wizard = {"active": False, "data": {}}
    
    #### 자연어/명령어 기반 예측 플로우
    did_predict = False

    # enable = st.session_state.get("enable_predict")
    bundle_loaded = st.session_state.get("bundle") is not None

    # 0) 번들 미로드/비활성 시 안내    
    intent_now = is_predict_intent(user_msg) or (parse_predict_message(user_msg) is not None) or (len(parse_predict_natural(user_msg)) > 0)
    if intent_now and not bundle_loaded:
        with st.chat_message("assistant"):
            st.warning("bundle 이 로드되지 않았습니다. 경로 확인후 **Load bundle**.")
        did_predict = True

    # 1) 마법사 진행 중이면 수집 계속    
    if not did_predict and st.session_state.predict_wizard["active"] and bundle_loaded:
        # CLI 메시지에는 자연어 파서를 적용하지 않아 값 덮어쓰기를 방지
        if user_msg.strip().lower().startswith("/predict"):
            d_cli = parse_predict_message(user_msg) or {}
            d_nat = {}
        else:
            d_cli = parse_predict_message(user_msg) or {}
            d_nat = parse_predict_natural(user_msg) or {}
        st.session_state.predict_wizard["data"].update({**d_nat, **d_cli})

        merged = {**d_nat, **d_cli}
        st.session_state.predict_wizard["data"].update(merged)
        # --- debug: 예측 입력 echo ---
        print("[DEBUG] predict_wizard step inputs:", merged)

        collected = st.session_state.predict_wizard["data"]
        missing = [k for k in REQUIRED_INPUT if k not in collected]
        if missing:
            with st.chat_message("assistant"):
                st.info(build_missing_prompt(missing))
            did_predict = True
        else:
            # 모든 입력 확보 → 예측 수행
            with st.chat_message("assistant"):
                try:
                    preds = st.session_state.bundle.predict_all(collected)
                    # LLM 을 이용한 자연어 요약
                    sys_prompt = (
                        "너는 구조공학 예측 결과를 한국어로 보고하는 도우미다. "
                        "주어진 수치를 변경하지 말고 그대로 사용해서 한 문단으로 간결히 설명하라."
                    )
                    user_prompt = (
                        "입력값과 예측값을 문장으로 요약하세요.\n"
                        f"- 입력: fck={collected.get('fck')} MPa, "
                        f"fy={collected.get('fy')} MPa, "
                        f"width={collected.get('width')} mm, "
                        f"height={collected.get('height')} mm, "
                        f"phi_mn={collected.get('phi_mn')} kN·m\n"
                        f"- 예측: Sm={preds.get('Sm')} mm², "
                        f"bd={preds.get('bd')} mm², "
                        f"rho={preds.get('rho')}, "
                        f"phi_mn={preds.get('phi_mn')} kN·m\n"
                        "형식 예: '단면계수 Sm은 (), 철근비 rho는 (), bd는 (), 공칭휨강도는 () 입니다'."
                    )
                    msgs = [
                        {'role':'system', 'content': sys_prompt},
                        {'role':'user',   'content': user_prompt},
                    ]
                    placeholder = st.empty()
                    visible, think = chat_once(llm, msgs, float(temp), float(topp), int(toks), placeholder)
                    placeholder.markdown(visible)
                    st.session_state.history.append({'role':'assistant', 'content': visible})
                except Exception as e:
                    st.error(f"예측 실패: {e}")
                    st.session_state.history.append({"role":"assistant","content": f"예측 실패: {e}"})
                finally:
                    st.session_state.predict_wizard = {"active": False, "data": {}}
            did_predict = True

    # 2) 새 입력으로 마법사 시작 여부 결정    
    if not did_predict and bundle_loaded:
        if user_msg.strip().lower().startswith("/predict"):
            d_cli0 = parse_predict_message(user_msg or "") or {}
            d_nat0 = {}
        else:
            d_cli0 = parse_predict_message(user_msg or "") or {}
            d_nat0 = parse_predict_natural(user_msg or "") or {}
        # trigger = (d_cli0 is not None) or (len(d_nat0) > 0)
        trigger = intent_now
        if trigger:
            # dict 병합: 자연어 > CLI 비어있을 경우 대비
            base = {}
            if isinstance(d_cli0, dict):
                base.update(d_cli0)
            base.update(d_nat0)
            
            # --- debug: 최초 예측 입력 echo ---
            print("[DEBUG] initial predict inputs:", base)
            missing = [k for k in REQUIRED_INPUT if k not in base]

            if missing:
                st.session_state.predict_wizard = {"active": True, "data": base}
                with st.chat_message("assistant"):
                    st.info(build_missing_prompt(missing))
                did_predict = True
            else:
                with st.chat_message('assistant'):
                    try:
                        preds = st.session_state.bundle.predict_all(base)
                        sys_prompt = (
                            "너는 구조공학 예측 결과를 한국어로 보고하는 도우미다. "
                            "주어진 수치를 변경하지 말고 그대로 한 문단으로 간결히 설명하라."
                        )
                        user_prompt = (
                            "입력값과 예측값을 문장으로 요약하세요.\n"
                            f"- 입력: fck={base.get('fck')} MPa, "
                            f"fy={base.get('fy')} MPa, "
                            f"width={base.get('width')} mm, "
                            f"height={base.get('height')} mm, "
                            f"phi_mn={base.get('phi_mn')} kN·m\n"
                            f"- 예측: Sm={preds.get('Sm')} mm², "
                            f"bd={preds.get('bd')} mm², "
                            f"rho={preds.get('rho')}, "
                            f"phi_mn={preds.get('phi_mn')} kN·m\n"
                            "형식 예: '단면계수 Sm은 (), 철근비 rho는 (), bd는 (), 공칭휨강도는 () 입니다'."
                        )
                        msgs = [
                            {'role':'system', 'content': sys_prompt},
                            {'role':'user',   'content': user_prompt},
                        ]
                        placeholder = st.empty()
                        visible, think = chat_once(llm, msgs, float(temp), float(topp), int(toks), placeholder)
                        placeholder.markdown(visible)
                        st.session_state.history.append({'role':'assistant', 'content': visible})
                    except Exception as e:
                        st.error(f'예측 실패: {e}')
                        st.session_state.history.append({"role":"assistant","content": f"예측 실패: {e}"})
                        
                did_predict = True

    # 모델 응답 (예측 플로우가 아니거나 종료된 경우)
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
