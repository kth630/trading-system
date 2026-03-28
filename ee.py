import sys
import os

def check_virtual_env():
    print("="*40)
    print("🔍 가상환경 및 Python 설정 확인")
    print("="*40)
    
    # 1. Python 실행 파일 경로 (현재 사용 중인 Python)
    print(f"📍 Python 실행 경로: {sys.executable}")
    
    # 2. Python 버전
    print(f"⚙️ Python 버전: {sys.version.split()[0]}")
    
    # 3. 가상환경 여부 확인
    # sys.prefix: 현재 실행 중인 환경의 경로
    # sys.base_prefix: 시스템에 설치된 기본 Python 경로 (가상환경 밖)
    is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if is_venv:
        print("✅ 현재 [가상환경] 🟢 활성화 상태입니다.")
        # 가상환경의 이름(폴더명) 추출
        venv_name = os.path.basename(sys.prefix)
        print(f"👉 가상환경 이름(경로): {venv_name} ({sys.prefix})")
    else:
        print("❌ 현재 가상환경이 🔴 아닙니다. (전역 시스템 Python 사용 중)")
    
    # 4. 패키지 설치 경로 (모듈을 어디서 가져오는지)
    print("-" * 40)
    print("📚 주요 패키지 참조 경로 (sys.path):")
    for path in sys.path[:5]: # 너무 많을 수 있으므로 상위 5개만 출력
        if path:
            print(f"  - {path}")
    print("="*40)

if __name__ == "__main__":
    check_virtual_env()
    
    # pandas가 정상적으로 임포트되는지 확인 (선택 사항)
    try:
        import pandas as pd
        print(f"✅ pandas 임포트 성공 (버전: {pd.__version__})")
        print(f"   경로: {pd.__file__}")
    except ImportError:
        print("❌ pandas 패키지를 찾을 수 없습니다. 현재 환경에 설치되어 있는지 확인하세요.")

import numpy as np
print(np.__version__)
print(np.__file__)
import yfinance as yf
print(yf.__version__)
print(yf.__file__)
import matplotlib.pyplot as plt
print(plt.__version__)
print(plt.__file__)
import plotly as py
print(py.__version__)
print(py.__file__)
import streamlit as st
print(st.__version__)
print(st.__file__)
import dotenv as dt
print(dt.__version__)
print(dt.__file__)