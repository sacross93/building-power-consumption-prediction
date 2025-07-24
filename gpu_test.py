#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 지원 테스트 스크립트
Author: Claude
Date: 2025-07-24
"""

import pandas as pd
import numpy as np

def test_gpu_support():
    """각 라이브러리의 GPU 지원 여부 확인"""
    
    print("=" * 60)
    print("GPU 지원 테스트")
    print("=" * 60)
    
    # 1. PyTorch CUDA 확인
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA 사용 가능: {cuda_available}")
        if cuda_available:
            print(f"  - GPU 개수: {torch.cuda.device_count()}")
            print(f"  - 현재 GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA 버전: {torch.version.cuda}")
            print(f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("PyTorch가 설치되지 않았습니다.")
        cuda_available = False
    
    # 2. LightGBM GPU 확인
    try:
        import lightgbm as lgb
        print(f"\nLightGBM 버전: {lgb.__version__}")
        
        # 간단한 데이터로 GPU 테스트
        X = np.random.rand(1000, 10)
        y = np.random.rand(1000)
        
        try:
            train_data = lgb.Dataset(X, label=y)
            params = {
                'objective': 'regression',
                'device_type': 'gpu',
                'verbosity': -1
            }
            model = lgb.train(params, train_data, num_boost_round=10)
            print("LightGBM GPU 지원: [SUCCESS]")
        except Exception as e:
            print(f"LightGBM GPU 지원: [FAILED] ({str(e)[:50]}...)")
            
    except ImportError:
        print("LightGBM이 설치되지 않았습니다.")
    
    # 3. XGBoost GPU 확인
    try:
        import xgboost as xgb
        print(f"\nXGBoost 버전: {xgb.__version__}")
        
        X = np.random.rand(1000, 10)
        y = np.random.rand(1000)
        
        try:
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0
            }
            model = xgb.train(params, dtrain, num_boost_round=10)
            print("XGBoost GPU 지원: [SUCCESS]")
        except Exception as e:
            print(f"XGBoost GPU 지원: [FAILED] ({str(e)[:50]}...)")
            
    except ImportError:
        print("XGBoost가 설치되지 않았습니다.")
    
    # 4. CatBoost GPU 확인
    try:
        import catboost as cb
        print(f"\nCatBoost 버전: {cb.__version__}")
        
        X = np.random.rand(1000, 10)
        y = np.random.rand(1000)
        
        try:
            model = cb.CatBoostRegressor(
                iterations=10,
                task_type='GPU',
                devices='0',
                verbose=False
            )
            model.fit(X, y)
            print("CatBoost GPU 지원: [SUCCESS]")
        except Exception as e:
            print(f"CatBoost GPU 지원: [FAILED] ({str(e)[:50]}...)")
            
    except ImportError:
        print("CatBoost가 설치되지 않았습니다.")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
    
    return cuda_available

def benchmark_cpu_vs_gpu():
    """CPU vs GPU 성능 벤치마크"""
    
    if not test_gpu_support():
        print("GPU를 사용할 수 없어 벤치마크를 건너뜁니다.")
        return
    
    print("\n" + "=" * 60)
    print("CPU vs GPU 성능 벤치마크")
    print("=" * 60)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples, n_features = 50000, 100
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    
    import time
    import lightgbm as lgb
    
    train_data = lgb.Dataset(X, label=y)
    base_params = {
        'objective': 'regression',
        'num_leaves': 127,
        'learning_rate': 0.1,
        'verbosity': -1
    }
    
    # CPU 테스트
    print("CPU 모드 테스트 중...")
    cpu_params = base_params.copy()
    cpu_params['device_type'] = 'cpu'
    
    start_time = time.time()
    cpu_model = lgb.train(cpu_params, train_data, num_boost_round=100)
    cpu_time = time.time() - start_time
    
    # GPU 테스트
    print("GPU 모드 테스트 중...")
    gpu_params = base_params.copy()
    gpu_params.update({
        'device_type': 'gpu',
        'max_bin': 127
    })
    
    try:
        start_time = time.time()
        gpu_model = lgb.train(gpu_params, train_data, num_boost_round=100)
        gpu_time = time.time() - start_time
        
        print(f"\n벤치마크 결과:")
        print(f"CPU 시간: {cpu_time:.2f}초")
        print(f"GPU 시간: {gpu_time:.2f}초")
        print(f"속도 향상: {cpu_time/gpu_time:.2f}배")
        
    except Exception as e:
        print(f"GPU 벤치마크 실패: {e}")

if __name__ == "__main__":
    test_gpu_support()
    benchmark_cpu_vs_gpu()