# chat_cli.py — llama-cpp-python CLI chatbot
# - llama-cpp-python 0.3.x 권장(0.2.90도 동작)
# - 기능: 역할 메시지, 스트리밍 출력, 대화 히스토리 저장/불러오기, 한국어 지원, 깨끗한 종료(llm.close)
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from llama_cpp import Llama

DEF_MODEL = r"c:\Users\BKHOME\mycode\LLM\models\Qwen3-1.7B-Q4_K_M.gguf"
#DEF_MODEL = r"c:\Users\BKHOME\mycode\LLM\models\Qwen3-4B-Q4_K_M.gguf"
#DEF_MODEL = r"c:\Users\BKHOME\mycode\LLM\models\EXAONE-Deep-2.4B.Q4_K_M.gguf"


BANNER = (
    "[READY] /q 종료  /new 새대화  /save [파일.json]  /load 파일.json  /sys [문구]\n"
    "        /temp 값  /toks 값  /ctx 값  /threads 값  /help 도움말"
)

# ---------- args ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEF_MODEL, help="GGUF model path")
    p.add_argument("--chat-format", default="auto", help="auto | qwen | llama-3 | ...")
    p.add_argument("--ctx", type=int, default=2048, help="context window size")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--topp", type=float, default=0.95)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument(
        "--system",
        default="당신은 한국어와 영어를 명확하고 간결하게 답하는 조수입니다.",
        help="system prompt",
    )
    return p.parse_args()

# ---------- helpers ----------
def auto_chat_format(model_path: str, user_arg: str) -> Optional[str]:
    """파일명으로 추정. 사용자가 지정하면 그대로 사용."""
    if user_arg and user_arg.lower() != "auto":
        return user_arg
    name = os.path.basename(model_path).lower()
    if "qwen" in name or "deepseek" in name:
        return "qwen"    # qwen2 미지원 환경 대비
    if "llama-3" in name or "llama3" in name:
        return "llama-3"
    return None

def load_llm(model_path: str, n_ctx: int, n_threads: int, chat_format: Optional[str]) -> Llama:
    """chat_format 미지원이면 None으로 자동 폴백."""
    try:
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            chat_format=chat_format,
            verbose=False,
        )
    except Exception as e:
        if "Invalid chat handler" in str(e):
            print("[WARN] chat_format unsupported → fallback to None")
            return Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                chat_format=None,
                verbose=False,
            )
        raise

def stream_answer(llm: Llama, messages: List[Dict[str, str]], temp: float, topp: float, toks: int) -> str:
    """스트리밍 우선, 불가하면 논스트림."""
    try:
        it = llm.create_chat_completion(
            messages=messages,
            temperature=temp,
            top_p=topp,
            max_tokens=toks,
            stream=True,
        )
        buf: List[str] = []
        for ch in it:
            delta = ch["choices"][0]["delta"].get("content", "")
            if delta:
                print(delta, end="", flush=True)
                buf.append(delta)
        print()
        return "".join(buf)
    except TypeError:
        resp = llm.create_chat_completion(
            messages=messages,
            temperature=temp,
            top_p=topp,
            max_tokens=toks,
        )
        txt = resp["choices"][0]["message"]["content"]
        print(txt)
        return txt

def save_history(path: str, system_msg: str, history: List[Dict[str, str]]) -> None:
    data = {"system": system_msg, "messages": history}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {path}")

def load_history(path: str) -> Tuple[str, List[Dict[str, str]]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d.get("system", "당신은 한국어와 영어를 명확하고 간결하게 답하는 조수입니다."), d.get("messages", [])

# ---------- main ----------
def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"[ERR] model not found: {args.model}")
        sys.exit(1)

    chat_format = auto_chat_format(args.model, args.chat_format)

    print("[INFO] Loading model…")
    llm: Optional[Llama] = None
    try:
        llm = load_llm(args.model, args.ctx, args.threads, chat_format)
        print("[INFO] Model loaded.")
        print(BANNER)

        system_msg = args.system
        history: List[Dict[str, str]] = []

        while True:
            try:
                s = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[EXIT]")
                break
            if not s:
                continue

            # commands
            if s in {"/q", "/quit", "/exit"}:
                print("[EXIT]")
                break

            if s == "/new":
                history.clear()
                print("[NEW] 대화 초기화")
                continue

            if s.startswith("/save"):
                parts = s.split(maxsplit=1)
                fname = parts[1] if len(parts) > 1 else f"chat_{int(time.time())}.json"
                try:
                    save_history(fname, system_msg, history)
                except Exception as e:
                    print("[ERR]", e)
                continue

            if s.startswith("/load"):
                parts = s.split(maxsplit=1)
                if len(parts) < 2:
                    print("사용법: /load 파일.json")
                    continue
                try:
                    system_msg, history = load_history(parts[1])
                    print(f"[LOADED] {parts[1]} ({len(history)} turns)")
                except Exception as e:
                    print("[ERR]", e)
                continue

            if s.startswith("/sys"):
                parts = s.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"[SYS] 현재: {system_msg}")
                else:
                    system_msg = parts[1]
                    print("[SYS] 변경 완료")
                continue

            if s.startswith("/temp"):
                try:
                    args.temp = float(s.split()[1])
                    print("[OK] temperature =", args.temp)
                except Exception:
                    print("사용법: /temp 0.6")
                continue

            if s.startswith("/toks"):
                try:
                    args.tokens = int(s.split()[1])
                    print("[OK] max_tokens =", args.tokens)
                except Exception:
                    print("사용법: /toks 512")
                continue

            if s.startswith("/ctx"):
                try:
                    args.ctx = int(s.split()[1])
                    print("[OK] n_ctx =", args.ctx)
                except Exception:
                    print("사용법: /ctx 2048")
                continue

            if s.startswith("/threads"):
                try:
                    args.threads = int(s.split()[1])
                    print("[OK] threads =", args.threads)
                except Exception:
                    print("사용법: /threads 6")
                continue

            if s == "/help":
                print(BANNER)
                continue

            # normal chat turn
            messages = [{"role": "system", "content": system_msg}, *history, {"role": "user", "content": s}]
            ans = stream_answer(llm, messages, args.temp, args.topp, args.tokens)
            history.append({"role": "user", "content": s})
            history.append({"role": "assistant", "content": ans})

    finally:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()