"""
개선된 XGBoost 학습: Method_08 최적화 적용

- Train Filtering (moderate: 84.8% 제거)
- Importance Sampling 가중치
- Test 적응형 피처 (49개 추가)
- Temperature-based CV
- GPU 가속
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE 계산"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    v = np.zeros_like(denom)
    v[mask] = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return 200.0 * np.mean(v)


def load_enhanced_data():
    """개선된 데이터 로드"""
    
    # 1. 필터링된 train 데이터 (moderate 전략)
    filtered_train_path = "eda/train_filtered_moderate.csv"
    if Path(filtered_train_path).exists():
        print(f"✅ 필터링된 train 데이터 로드: {filtered_train_path}")
        filtered_train = pd.read_csv(filtered_train_path, encoding='utf-8-sig')
        print(f"   필터링된 크기: {len(filtered_train):,}행")
    else:
        print("❌ 필터링된 데이터 없음 - 원본 사용")
        filtered_train = pd.read_csv("../method_07/train_building_merged.csv", encoding='utf-8-sig')
    
    # 2. 적응형 피처가 추가된 데이터
    enhanced_train_path = "eda/train_test_adaptive_features.csv"
    enhanced_test_path = "eda/test_test_adaptive_features.csv"
    
    if Path(enhanced_train_path).exists() and Path(enhanced_test_path).exists():
        print(f"✅ 적응형 피처 데이터 로드")
        enhanced_train = pd.read_csv(enhanced_train_path, encoding='utf-8-sig')
        enhanced_test = pd.read_csv(enhanced_test_path, encoding='utf-8-sig')
        print(f"   Enhanced train: {len(enhanced_train):,}행 × {len(enhanced_train.columns)}열")
        print(f"   Enhanced test: {len(enhanced_test):,}행 × {len(enhanced_test.columns)}열")
    else:
        print("❌ 적응형 피처 데이터 없음 - 원본 사용")
        enhanced_train = pd.read_csv("../method_07/train_building_merged.csv", encoding='utf-8-sig')
        enhanced_test = pd.read_csv("../method_07/test_building_merged.csv", encoding='utf-8-sig')
    
    # 3. 필터링 + 적응형 피처 결합
    # 필터링된 인덱스와 적응형 피처를 매칭
    if Path(filtered_train_path).exists() and Path(enhanced_train_path).exists():
        # 필터링된 인덱스 추출
        original_train = pd.read_csv("../method_07/train_building_merged.csv", encoding='utf-8-sig')
        
        # 필터링된 데이터와 매칭되는 행들을 적응형 피처 데이터에서 선택
        if 'num_date_time' in filtered_train.columns and 'num_date_time' in enhanced_train.columns:
            filtered_keys = set(filtered_train['num_date_time'])
            mask = enhanced_train['num_date_time'].isin(filtered_keys)
            final_train = enhanced_train[mask].copy()
            print(f"✅ 필터링 + 적응형 피처 결합: {len(final_train):,}행")
        else:
            # 인덱스 기반 매칭
            final_train = enhanced_train.iloc[filtered_train.index].copy()
            print(f"✅ 인덱스 기반 결합: {len(final_train):,}행")
    else:
        final_train = enhanced_train
    
    # 4. Importance Sampling 가중치 로드
    weights_path = "eda/importance_sampling_weights.csv"
    sample_weights = None
    
    if Path(weights_path).exists():
        print(f"✅ Importance Sampling 가중치 로드")
        weights_df = pd.read_csv(weights_path, encoding='utf-8-sig')
        
        # 가중치를 final_train과 매칭
        if len(weights_df) == len(final_train):
            sample_weights = weights_df['combined_weights'].values
        elif 'index' in weights_df.columns:
            # 인덱스 기반 매칭
            weight_dict = dict(zip(weights_df['index'], weights_df['combined_weights']))
            sample_weights = np.array([weight_dict.get(i, 1.0) for i in final_train.index])
        else:
            sample_weights = None
        
        if sample_weights is not None:
            print(f"   가중치 통계: 평균 {sample_weights.mean():.3f}, 범위 {sample_weights.min():.3f}~{sample_weights.max():.3f}")
    else:
        print("❌ 가중치 파일 없음 - 균등 가중치 사용")
    
    return final_train, enhanced_test, sample_weights


def build_features(df: pd.DataFrame, target_col: str = '전력소비량(kWh)') -> List[str]:
    """피처 선택"""
    drop_cols = {
        "일시", "num_date_time", "건물번호", "건물유형", target_col, "log_power",
        "datetime", "date", "month", "hour", "weekday", "day"  # 중간 생성 컬럼들
    }
    
    # 숫자형 컬럼만 선택
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in num_cols if c not in drop_cols]
    
    # 상수 컬럼 제거
    features = [c for c in features if df[c].nunique(dropna=True) > 1]
    
    print(f"선택된 피처: {len(features)}개")
    return features


class TemperatureBasedCV:
    """Temperature 기반 CV (간단 구현)"""
    
    def __init__(self, n_splits=5, random_state=2025):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def split(self, X, y=None):
        """온도 기준 분할"""
        
        if '기온(°C)' not in X.columns:
            # Fallback: 랜덤 분할
            np.random.seed(self.random_state)
            indices = np.random.permutation(len(X))
            fold_size = len(X) // self.n_splits
            
            for i in range(self.n_splits):
                start = i * fold_size
                end = (i + 1) * fold_size if i < self.n_splits - 1 else len(X)
                val_indices = indices[start:end]
                train_indices = np.concatenate([indices[:start], indices[end:]])
                yield train_indices, val_indices
            return
        
        # 온도 기준 분할
        temp_values = X['기온(°C)'].values
        
        # Test 온도 분포 추정 (여름철 기준)
        test_temp_mean = 27.0  # 8월 평균 추정
        test_temp_std = 3.0
        
        # 각 샘플의 test 유사도 계산
        temp_similarity = np.exp(-np.abs(temp_values - test_temp_mean) / test_temp_std)
        
        np.random.seed(self.random_state)
        
        for fold in range(self.n_splits):
            # 높은 유사도 샘플을 validation에 우선 배치
            similarity_ranks = np.argsort(-temp_similarity)  # 내림차순
            
            fold_size = len(X) // self.n_splits
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.n_splits - 1 else len(X)
            
            # 유사도 높은 순서대로 validation 선택
            val_indices = similarity_ranks[start_idx:end_idx]
            train_indices = np.setdiff1d(np.arange(len(X)), val_indices)
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def train_xgb_with_improvements(X_train, y_train, X_test, sample_weights=None, 
                              cv_strategy='temperature_based', use_gpu=False):
    """개선된 XGBoost 학습"""
    
    from xgboost import XGBRegressor
    
    print(f"\n🚀 개선된 XGBoost 학습 시작")
    print(f"   Train 크기: {len(X_train):,} × {len(X_train.columns)}")
    print(f"   Test 크기: {len(X_test):,} × {len(X_test.columns)}")
    print(f"   Sample weights: {'적용' if sample_weights is not None else '미적용'}")
    print(f"   CV 전략: {cv_strategy}")
    print(f"   GPU 사용: {use_gpu}")
    
    # XGBoost 파라미터 (Test 과적합 방지 최적화)
    xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.02,  # 더 보수적
        'n_estimators': 3000,
        'max_depth': 5,  # 복잡도 감소
        'min_child_weight': 10,  # 더 보수적
        'subsample': 0.7,  # 더 강한 서브샘플링
        'colsample_bytree': 0.7,
        'reg_lambda': 3.0,  # 강한 정규화
        'reg_alpha': 0.5,
        'tree_method': 'hist',
        'random_state': 2025,
        'verbosity': 0,
        'n_jobs': -1
    }
    
    if use_gpu:
        xgb_params.update({
            'device': 'cuda',
            'tree_method': 'hist'
        })
        print("🔥 GPU 가속 활성화")
    
    # CV 설정
    if cv_strategy == 'temperature_based':
        cv = TemperatureBasedCV(n_splits=5)
        print("🌡️ Temperature-based CV 사용")
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=2025)
        print("📋 Standard KFold CV 사용")
    
    # CV 학습
    cv_scores = []
    test_predictions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        print(f"\n   폴드 {fold_idx + 1}/5 학습 중...")
        
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # 가중치 적용
        fold_weights = sample_weights[train_idx] if sample_weights is not None else None
        
        # 모델 학습
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_fold_train, y_fold_train,
            sample_weight=fold_weights,
            eval_set=[(X_fold_val, y_fold_val)],
            early_stopping_rounds=200,
            verbose=False
        )
        
        # 검증 성능
        val_pred_log = model.predict(X_fold_val)
        val_pred = np.expm1(val_pred_log)
        val_true = np.expm1(y_fold_val)
        
        # 음수 클리핑
        val_pred = np.maximum(val_pred, 0)
        
        fold_smape = smape(val_true, val_pred)
        cv_scores.append(fold_smape)
        
        print(f"      폴드 {fold_idx + 1} SMAPE: {fold_smape:.3f}%")
        
        # Test 예측
        test_pred_log = model.predict(X_test)
        test_pred = np.expm1(test_pred_log)
        test_pred = np.maximum(test_pred, 0)  # 음수 클리핑
        test_predictions.append(test_pred)
    
    # 최종 결과
    final_test_pred = np.mean(test_predictions, axis=0)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\n✅ 학습 완료!")
    print(f"   CV SMAPE: {cv_mean:.3f}% ± {cv_std:.3f}%")
    print(f"   개별 폴드: {', '.join([f'{s:.3f}%' for s in cv_scores])}")
    
    return final_test_pred, cv_scores


def apply_post_processing(predictions: np.ndarray, test_data: pd.DataFrame) -> np.ndarray:
    """간단한 후처리"""
    
    print("\n🔧 후처리 적용...")
    
    processed = predictions.copy()
    
    # 1. 극단값 클리핑 (99.5% percentile)
    upper_bound = np.percentile(processed, 99.5)
    processed = np.minimum(processed, upper_bound)
    
    # 2. 음수 제거
    processed = np.maximum(processed, 0)
    
    # 3. 건물별 합리적 범위 (있는 경우)
    if '건물번호' in test_data.columns:
        for building_id in test_data['건물번호'].unique():
            building_mask = test_data['건물번호'] == building_id
            building_preds = processed[building_mask]
            
            # 건물별 상한선 (중앙값의 10배)
            building_median = np.median(building_preds)
            building_upper = building_median * 10
            
            processed[building_mask] = np.minimum(building_preds, building_upper)
    
    changes = np.sum(processed != predictions)
    print(f"   수정된 예측값: {changes:,}개")
    
    return processed


def main():
    """메인 실행"""
    
    parser = argparse.ArgumentParser(description="개선된 XGBoost 학습")
    parser.add_argument("--cv-strategy", type=str, default="temperature_based",
                       choices=["temperature_based", "standard"],
                       help="CV 전략")
    parser.add_argument("--use-gpu", action="store_true", help="GPU 사용")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    print("🎯 Method_08 개선된 XGBoost 학습")
    print("="*50)
    
    # 1. 데이터 로드
    print("\n📁 데이터 로드...")
    train_data, test_data, sample_weights = load_enhanced_data()
    
    # 2. 피처 준비
    print("\n🔧 피처 준비...")
    target_col = '전력소비량(kWh)'
    
    if target_col not in train_data.columns:
        raise ValueError(f"타겟 컬럼 '{target_col}' 없음")
    
    features = build_features(train_data, target_col)
    
    # Test와 공통 피처만 사용
    common_features = [f for f in features if f in test_data.columns]
    print(f"Train-Test 공통 피처: {len(common_features)}개")
    
    # 데이터 준비
    X_train = train_data[common_features].fillna(0)
    y_train = np.log1p(train_data[target_col].fillna(0))
    X_test = test_data[common_features].fillna(0)
    
    # 3. 모델 학습
    print("\n🚀 모델 학습...")
    test_predictions, cv_scores = train_xgb_with_improvements(
        X_train, y_train, X_test, 
        sample_weights=sample_weights,
        cv_strategy=args.cv_strategy,
        use_gpu=args.use_gpu
    )
    
    # 4. 후처리
    final_predictions = apply_post_processing(test_predictions, test_data)
    
    # 5. 결과 저장
    print(f"\n💾 결과 저장...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 예측 결과
    test_with_pred = test_data.copy()
    test_with_pred['prediction'] = final_predictions
    test_with_pred.to_csv(output_dir / "test_predictions_improved.csv", 
                         index=False, encoding='utf-8-sig')
    
    # 간단한 제출 파일
    submission = pd.DataFrame({
        'id': range(len(final_predictions)),
        'prediction': final_predictions
    })
    submission.to_csv(output_dir / "submission_improved.csv", 
                     index=False, encoding='utf-8-sig')
    
    # 결과 요약
    results_summary = {
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'train_size': len(X_train),
        'features_count': len(common_features),
        'sample_weights_used': sample_weights is not None,
        'cv_strategy': args.cv_strategy,
        'gpu_used': args.use_gpu,
        'prediction_stats': {
            'mean': float(np.mean(final_predictions)),
            'std': float(np.std(final_predictions)),
            'min': float(np.min(final_predictions)),
            'max': float(np.max(final_predictions))
        }
    }
    
    import json
    with open(output_dir / "training_summary.json", 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 결과 저장 완료: {output_dir}/")
    print(f"   - test_predictions_improved.csv: 상세 예측 결과")
    print(f"   - submission_improved.csv: 제출용 파일")
    print(f"   - training_summary.json: 학습 요약")
    
    print(f"\n🎯 최종 성능: CV SMAPE {np.mean(cv_scores):.3f}% ± {np.std(cv_scores):.3f}%")
    
    # Method_07 대비 비교 (참고용)
    method07_smape = 1.56  # Internal validation 결과
    improvement = method07_smape - np.mean(cv_scores)
    
    print(f"\n📊 Method_07 대비:")
    print(f"   Method_07 CV: ~{method07_smape}%")
    print(f"   Method_08 CV: {np.mean(cv_scores):.3f}%")
    if improvement > 0:
        print(f"   개선: {improvement:.3f}%p 향상 🎉")
    else:
        print(f"   변화: {improvement:.3f}%p")
    
    return results_summary


if __name__ == "__main__":
    results = main()