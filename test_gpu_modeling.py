#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 모델링 테스트 스크립트
Author: Claude
Date: 2025-07-24
"""

from modeling_claude import PowerConsumptionPredictor
import time

def test_gpu_modeling():
    """GPU 모델링 기능 테스트"""
    
    print("=" * 60)
    print("GPU 모델링 테스트")
    print("=" * 60)
    
    # GPU 모델 초기화
    predictor = PowerConsumptionPredictor(
        model_type='lightgbm', 
        random_state=42, 
        use_gpu=True
    )
    
    try:
        # 데이터 로드
        print("\n1. 데이터 로드 중...")
        predictor.load_processed_data('none_log')
        
        # 피처 준비
        print("2. 피처 준비 중...")
        predictor.prepare_features()
        
        # 간단한 하이퍼파라미터 테스트 (2 trials만)
        print("3. GPU 하이퍼파라미터 최적화 테스트...")
        start_time = time.time()
        predictor.optimize_hyperparameters(n_trials=2, cv_folds=2)
        gpu_time = time.time() - start_time
        
        print(f"GPU 최적화 시간: {gpu_time:.2f}초")
        print(f"최적 SMAPE: {predictor.best_score:.4f}")
        
        print("\n[SUCCESS] GPU 모델링 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"[ERROR] GPU 모델링 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_gpu_modeling()