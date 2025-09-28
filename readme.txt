
1) 온라인 PC에서 설치

핵심 개념:
휠(wheel, .whl): 파이썬 패키지의 설치 파일.
오프라인 설치를 위해 필요한 모든 의존 패키지를 한 폴더에 미리 내려받는다.
**온라인·오프라인 PC의 Python 버전(예: 3.12)과 아키텍처(64-bit)**가 같아야 한다.

powershell
py -0p	# 설치된 python 목록

가상환경을 생성,활성화
vscode 에서 폴더를 하나 생성하고 그 안에서

powershell
python -m venv .venv 또는 py -3.12 -m venv .venvllm
.\.venv\Scripts\activate.ps1

온라인 환경인 경우:
pip install "llama-cpp-python==0.3.16" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
로 설치,  빌드된 whl 파일이 생성된다

CUDA GPU 지원할 경우:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

CUDA 지원 wheel을 설치하기 위해 시스템에 NVIDIA GPU와 적절한 CUDA Toolkit이 설치되어 있어야 합니다.
설치 중 wheel이 없는 경우, pip은 소스 빌드를 시도할 수 있으므로
CUDA 관련 환경 변수(예: GGML_CUDA=on)와 빌드 도구(CMake, Ninja 등)가 필요할 수 있습니다.


2) 온라인 PC에서 설치오프라인 설치용 파일 준비

#--------------------------------------------------------------------------------------------------
# 오프라인 설치를 위해 Windows + Python 3.12 환경에서 llama-cpp-python을 빌드하고 wheel 파일 생성
#--------------------------------------------------------------------------------------------------

# Windows 에 프로그램 설치
winget install --id=Git.Git -e
winget install --id=Kitware.CMake -e
winget install --id=Ninja-build.Ninja -e
vs_BuildTools.exe		# VS Build Tools 설치

powershell 에서
mkdir C:\build\llama && cd C:\build\llama
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install build setuptools wheel cmake ninja

vscode의 git bash 창을 열고
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd \llama-cpp-python
git checkout v0.3.16	# 원하는 태그
git submodule update --init --recursive		# 생략하면 error 발생
# (패키지 버전을 예를 들어 0.2.90 으로 변경할 경우, llama-cpp-python 폴더 삭제 후 위 과정 반복)

vscode powershell 에서
setx GGML_CUDA 0			# CUDA 비활성
$env:GGML_CUDA="0"			# 새 PowerShell 창이 아니면 현재 세션에도 반영
cd llama-cpp-python			# 최상위 폴더로 이동
python -m build --wheel		# BUILD 시작

여러 python 버전이 섞여있을때
py -3.12 -m pip install build setuptools wheel cmake ninja
py -3.12 -m build --wheel

#### 생성 위치: llama-cpp-python\dist\*.whl

2) 오프라인 설치 방법

powershell 에서 로컬 파일 설치(오프라인):
pip install --no-index --find-links=[deps 폴더] [llama-cpp-python whl]
pip install --no-index --find-links=./deps llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl (use --force-reinstall if needed)

설치 확인:
pip show llama-cpp-python

#------------------------
# chat_app.py 실행 방법
#------------------------

in command window:
.\.venvllm\Scripts\activate(.bat)

in powershell:
.\.venvllm\Scripts\activate.ps1

in powershell, run:
streamlit run chat_app_V2.1.py

input:
/predict fck=27, fy=400, width=800, height=1000, mu=2000
콘크리트 단면의 정보는 fck=27, 철근강도 fy=400 MPa, 단면폭=800mm, 단면높이=1000mm, mu=2000 kN-m 가 작용할때 철근비와 단면공칭휨강도는?