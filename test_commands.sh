#!/bin/bash

# GPU 지원 모델 테스트 명령어 스크립트
echo "🚀 GPU 지원 모델 테스트 명령어들"
echo "================================="

echo ""
echo "1. 💻 CPU 버전 (기본) - 빠른 테스트"
echo "uv run test4_optuna_gpu.py --n-trials 10"

echo ""
echo "2. 🔥 GPU 버전 - 빠른 테스트"
echo "uv run test4_optuna_gpu.py --gpu --n-trials 10"

echo ""
echo "3. 💻 CPU 버전 - 정밀 튜닝"
echo "uv run test4_optuna_gpu.py --n-trials 30"

echo ""
echo "4. 🔥 GPU 버전 - 정밀 튜닝"
echo "uv run test4_optuna_gpu.py --gpu --n-trials 30"

echo ""
echo "5. 🔥 GPU 버전 - 고성능 튜닝"
echo "uv run test4_optuna_gpu.py --gpu --n-trials 50"

echo ""
echo "6. 🔥 GPU 버전 - 최고 성능 도전"
echo "uv run test4_optuna_gpu.py --gpu --n-trials 100"

echo ""
echo "================================="
echo "📋 사용법:"
echo "- GPU가 없으면 자동으로 CPU로 fallback"
echo "- --n-trials 값이 클수록 더 좋은 성능, 더 오래 걸림"
echo "- submission.csv는 매 실행마다 덮어쓰기됨"
echo "- 실행 중 GPU 메모리 부족 시 --n-trials 값을 줄여보세요"

# GPU 정보 확인 명령어
echo ""
echo "🔍 GPU 정보 확인:"
echo "nvidia-smi"
echo "python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""

# 메모리 모니터링
echo ""
echo "📊 실행 중 모니터링:"
echo "watch -n 1 nvidia-smi  # GPU 메모리 실시간 모니터링"
echo "htop                   # CPU/RAM 모니터링" 