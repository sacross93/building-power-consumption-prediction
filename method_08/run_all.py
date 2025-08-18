#!/usr/bin/env python3
"""
Method 08 전체 파이프라인 순차 실행기

모든 단계를 순서대로 실행하여 Test 과적합 최적화를 수행합니다.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_step(step_name: str, script_name: str):
    """개별 스텝 실행"""
    print(f"\n{'='*60}")
    print(f"{step_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name], 
            cwd=Path(__file__).parent,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ {step_name} 완료 (소요시간: {elapsed:.1f}초)")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {step_name} 실패 (소요시간: {elapsed:.1f}초)")
        print(f"오류: {e}")
        return False

def main():
    """전체 파이프라인 실행"""
    
    print("🚀 Method 08: Test 과적합 최적화 파이프라인")
    print("목표: Train SMAPE 2% → Test SMAPE 6.8% → 4% 이하 개선")
    
    total_start = time.time()
    
    # 실행 단계 정의
    steps = [
        ("1단계: Train vs Test 분포 분석", "01_distribution_analysis.py"),
        ("2단계: Importance Sampling 가중치 계산", "02_importance_sampling.py"),
        ("3단계: Train 데이터 필터링", "03_train_filtering.py"),
        ("4단계: Test 적응형 피처 엔지니어링", "04_test_adaptive_features.py"),
        ("5단계: CV 전략 분석", "05_cv_strategy.py"),
        ("6단계: Post-Processing 테스트", "06_post_processing.py"),
    ]
    
    # 단계별 실행
    success_count = 0
    
    for step_name, script_name in steps:
        success = run_step(step_name, script_name)
        if success:
            success_count += 1
        else:
            print(f"\n❌ {step_name} 실패로 인해 파이프라인을 중단합니다.")
            break
    
    # 최종 결과
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("최종 결과")
    print(f"{'='*60}")
    
    print(f"성공한 단계: {success_count}/{len(steps)}")
    print(f"전체 소요시간: {total_elapsed:.1f}초")
    
    if success_count == len(steps):
        print("\n🎉 모든 단계 성공!")
        print("\n📊 주요 결과:")
        print("   - Train 분포 분석: eda/distribution_analysis_report.txt")
        print("   - 가중치 계산: eda/importance_sampling_weights.csv")
        print("   - 필터링 결과: eda/train_filtered_*.csv") 
        print("   - 적응형 피처: eda/train_test_adaptive_features.csv")
        print("   - CV 전략: eda/cv_strategy_analysis_report.txt")
        print("   - 후처리 분석: eda/post_processing_report.txt")
        
        print("\n🎯 다음 단계:")
        print("   1. eda/ 디렉토리에서 각 분석 결과 확인")
        print("   2. 필터링된 데이터로 실제 모델 학습")
        print("   3. Test 성능 확인 및 추가 조정")
        
        print("\n💡 권장 설정:")
        print("   - 필터링 전략: moderate (84.8% 제거)")
        print("   - CV 전략: temperature_based")
        print("   - 가중치 활용: geometric_mean 조합")
        
    else:
        print(f"\n❌ 일부 단계 실패 ({success_count}/{len(steps)})")
        print("   실패한 단계를 개별적으로 재실행하세요.")
    
    return success_count == len(steps)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)