#!/bin/bash

echo "🚀 Method_08 GPU 서버 실행 스크립트"
echo "=================================="

# 1. 환경 확인
echo "📋 환경 확인..."
# uv run을 통해 파이썬 버전을 확인합니다.
uv run python --version
echo "GPU 상태:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "NVIDIA GPU 없음"

# 2. 빠른 설정 실행 (데이터 준비)
# 이 스크립트는 전처리 단계를 실행합니다.
echo ""
echo "⚙️ 데이터 준비..."
uv run python quick_setup.py

# 3. GPU 학습 실행
# 이 스크립트는 준비된 데이터로 모델을 학습합니다.
echo ""
echo "🔥 GPU 학습 시작..."
uv run python train_with_improvements.py \
    --use-gpu \
    --cv-strategy temperature_based \
    --output-dir results_gpu

echo ""
echo "✅ 완료! results_gpu/ 디렉토리에서 결과 확인"
