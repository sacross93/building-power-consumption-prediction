"""
GPU 서버 빠른 설정 스크립트

Method_08 파이프라인을 실행해서 필요한 데이터를 생성합니다.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """명령어 실행"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 실패: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def main():
    """빠른 설정 실행"""
    
    print("🚀 Method_08 GPU 서버 빠른 설정")
    print("="*50)
    
    # 필요한 디렉토리 확인
    if not Path("../method_07").exists():
        print("❌ method_07 디렉토리가 없습니다.")
        print("   git clone으로 전체 프로젝트를 받았는지 확인하세요.")
        return False
    
    # 단계별 실행
    steps = [
        ("python 01_distribution_analysis.py", "분포 분석"),
        ("python 02_importance_sampling.py", "가중치 계산"), 
        ("python 03_train_filtering.py", "Train 필터링"),
        ("python 04_test_adaptive_features.py", "피처 엔지니어링"),
    ]
    
    success_count = 0
    for cmd, desc in steps:
        success = run_command(cmd, desc)
        if success:
            success_count += 1
        else:
            print(f"\n❌ {desc} 단계에서 실패했습니다.")
            break
    
    if success_count == len(steps):
        print(f"\n🎉 모든 설정 완료! ({success_count}/{len(steps)})")
        print("\n📁 생성된 파일들:")
        
        files_to_check = [
            "eda/train_filtered_moderate.csv",
            "eda/importance_sampling_weights.csv", 
            "eda/train_test_adaptive_features.csv",
            "eda/test_test_adaptive_features.csv"
        ]
        
        for file_path in files_to_check:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size / 1024 / 1024  # MB
                print(f"   ✅ {file_path} ({size:.1f} MB)")
            else:
                print(f"   ❌ {file_path} (없음)")
        
        print(f"\n🚀 이제 다음 명령어로 GPU 학습을 시작하세요:")
        print(f"   python train_with_improvements.py --use-gpu --cv-strategy temperature_based")
        
    else:
        print(f"\n❌ 설정 실패 ({success_count}/{len(steps)})")
        print("   개별 스크립트를 직접 실행해보세요.")
    
    return success_count == len(steps)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)