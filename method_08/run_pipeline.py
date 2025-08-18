"""
Integrated Test-Overfitting Pipeline: 통합 Test 과적합 최적화 파이프라인

지금까지 구현한 모든 전략을 통합하여 Train-Test gap을 최소화하고
Test 성능을 극대화하는 완전한 파이프라인을 제공합니다.

Pipeline Steps:
1. Distribution Analysis
2. Train Filtering  
3. Importance Sampling
4. Test-Adaptive Feature Engineering
5. Test-Similar CV Strategy
6. Model Training with Adaptations
7. Post-Processing
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 직접 실행으로 대체
def run_distribution_analysis(train_path, test_path):
    """분포 분석 실행"""
    import subprocess
    result = subprocess.run([sys.executable, "01_distribution_analysis.py"], 
                          capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.returncode == 0

def run_importance_sampling():
    """가중치 계산 실행"""
    import subprocess
    result = subprocess.run([sys.executable, "02_importance_sampling.py"], 
                          capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.returncode == 0

def run_train_filtering():
    """필터링 실행"""
    import subprocess
    result = subprocess.run([sys.executable, "03_train_filtering.py"], 
                          capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.returncode == 0

def run_feature_engineering():
    """피처 엔지니어링 실행"""
    import subprocess
    result = subprocess.run([sys.executable, "04_test_adaptive_features.py"], 
                          capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.returncode == 0

def run_cv_analysis():
    """CV 분석 실행"""
    import subprocess
    result = subprocess.run([sys.executable, "05_cv_strategy.py"], 
                          capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.returncode == 0

class TestOverfittingPipeline:
    """Test 과적합 최적화 통합 파이프라인"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config
        self.results = {}
        self.save_dir = Path(config.get('save_dir', 'results'))
        self.save_dir.mkdir(exist_ok=True)
        
        # 컴포넌트 초기화
        self.distribution_analyzer = None
        self.importance_sampler = None
        self.train_filter = None
        self.feature_engineer = None
        self.cv_strategy = None
        self.post_processor = None
        
        print(f"🚀 Test 과적합 최적화 파이프라인 초기화")
        print(f"   설정: {config}")
        print(f"   결과 저장: {self.save_dir}")
    
    def load_and_analyze_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """1단계: 데이터 로드 및 분포 분석"""
        
        print("\n" + "="*60)
        print("1단계: 데이터 로드 및 분포 분석")
        print("="*60)
        
        # 데이터 로드
        train, test = load_data(train_path, test_path)
        
        print(f"데이터 로드 완료:")
        print(f"  Train: {len(train):,}행 × {len(train.columns)}열")
        print(f"  Test: {len(test):,}행 × {len(test.columns)}열")
        
        # 분포 분석
        shift_metrics = calculate_distribution_shift_metrics(train, test)
        
        # 분포 차이 요약
        significant_shifts = []
        for col, metrics in shift_metrics.items():
            if metrics['ks_p_value'] < 0.05:
                significant_shifts.append(f"{col}: KS={metrics['ks_statistic']:.3f}")
        
        print(f"\n📊 분포 차이 분석:")
        print(f"  유의한 분포 차이: {len(significant_shifts)}개 변수")
        for shift in significant_shifts:
            print(f"    - {shift}")
        
        self.results['data_analysis'] = {
            'train_shape': train.shape,
            'test_shape': test.shape,
            'distribution_shifts': shift_metrics,
            'significant_shifts': significant_shifts
        }
        
        return train, test
    
    def filter_train_data(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """2단계: Train 데이터 필터링"""
        
        print("\n" + "="*60)
        print("2단계: Train 데이터 필터링")
        print("="*60)
        
        filter_strategy = self.config.get('filter_strategy', 'moderate')
        
        filtered_train, filter_results = apply_combined_filter(
            train, test, 
            filter_strategy=filter_strategy,
            save_analysis=True
        )
        
        removal_rate = filter_results['removal_rate']
        print(f"\n🗂️ 필터링 결과:")
        print(f"  제거율: {removal_rate:.1f}%")
        print(f"  남은 데이터: {len(filtered_train):,}행")
        
        self.results['filtering'] = filter_results
        
        return filtered_train
    
    def calculate_sample_weights(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """3단계: Importance Sampling 가중치 계산"""
        
        print("\n" + "="*60)
        print("3단계: Importance Sampling 가중치 계산")
        print("="*60)
        
        # 기후 변수 기반 가중치
        sampler = ImportanceSampler(method='distance_based')
        sampler.fit(train, test)
        weather_weights = sampler.calculate_weights(train)
        
        # 계절적 가중치
        seasonal_weights = calculate_seasonal_weights(train, test)
        
        # 건물 특성 가중치  
        building_weights = calculate_building_weights(train, test)
        
        # 가중치 조합
        all_weights = [weather_weights, seasonal_weights, building_weights]
        weight_names = ['weather', 'seasonal', 'building']
        
        combined_weights = combine_weights(all_weights, weight_names, 'geometric_mean')
        
        print(f"\n⚖️ 가중치 계산 결과:")
        print(f"  평균: {combined_weights.mean():.3f}")
        print(f"  표준편차: {combined_weights.std():.3f}")
        print(f"  범위: {combined_weights.min():.3f} ~ {combined_weights.max():.3f}")
        
        self.results['importance_sampling'] = {
            'weights_stats': {
                'mean': combined_weights.mean(),
                'std': combined_weights.std(),
                'min': combined_weights.min(),
                'max': combined_weights.max()
            }
        }
        
        return combined_weights
    
    def engineer_features(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """4단계: Test 적응형 피처 엔지니어링"""
        
        print("\n" + "="*60)
        print("4단계: Test 적응형 피처 엔지니어링")
        print("="*60)
        
        enhanced_train, enhanced_test, feature_stats = apply_test_adaptive_engineering(
            train, test, save_analysis=True
        )
        
        print(f"\n🔧 피처 엔지니어링 결과:")
        print(f"  추가된 피처: {feature_stats['total_new_features']}개")
        print(f"  최종 피처: {feature_stats['final_features']}개")
        
        for group, features in feature_stats['feature_groups'].items():
            if features:
                print(f"    - {group}: {len(features)}개")
        
        self.results['feature_engineering'] = feature_stats
        
        return enhanced_train, enhanced_test
    
    def setup_cv_strategy(self, train: pd.DataFrame, test: pd.DataFrame):
        """5단계: CV 전략 설정"""
        
        print("\n" + "="*60)
        print("5단계: CV 전략 설정")
        print("="*60)
        
        cv_type = self.config.get('cv_strategy', 'test_similarity')
        n_splits = self.config.get('cv_splits', 5)
        
        if cv_type == 'test_similarity':
            self.cv_strategy = TestSimilarityCV(
                n_splits=n_splits,
                test_similarity_ratio=0.7,
                random_state=2025
            )
            self.cv_strategy.set_test_reference(test)
            
        elif cv_type in ['temperature_based', 'season_based', 'clustering_based']:
            self.cv_strategy = ClimateBasedCV(
                n_splits=n_splits,
                climate_strategy=cv_type,
                random_state=2025
            )
            self.cv_strategy.set_test_reference(test)
            
        else:
            # Fallback: 시간 기반
            from sklearn.model_selection import TimeSeriesSplit
            self.cv_strategy = TimeSeriesSplit(n_splits=n_splits)
        
        print(f"\n📋 CV 전략 설정:")
        print(f"  전략: {cv_type}")
        print(f"  폴드 수: {n_splits}")
        
        self.results['cv_setup'] = {
            'strategy': cv_type,
            'n_splits': n_splits
        }
    
    def train_model(self, train: pd.DataFrame, test: pd.DataFrame, 
                   sample_weights: np.ndarray, target_col: str = '전력소비량(kWh)') -> Tuple[np.ndarray, Dict]:
        """6단계: 모델 학습"""
        
        print("\n" + "="*60)
        print("6단계: 적응형 모델 학습")
        print("="*60)
        
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_absolute_error
        
        # 피처 준비
        drop_cols = {'일시', 'num_date_time', '건물번호', '건물유형', target_col, 'log_power', 'datetime', 'date'}
        
        feature_cols = [col for col in train.columns 
                       if col not in drop_cols and train[col].dtype in ['int64', 'float64']]
        
        # Test와 공통 피처만 사용
        feature_cols = [col for col in feature_cols if col in test.columns]
        
        X_train = train[feature_cols].fillna(0)
        y_train = np.log1p(train[target_col].fillna(0))
        X_test = test[feature_cols].fillna(0)
        
        print(f"모델 학습 설정:")
        print(f"  피처 수: {len(feature_cols)}")
        print(f"  Train 크기: {len(X_train):,}")
        print(f"  Test 크기: {len(X_test):,}")
        print(f"  Sample weights 적용: {'Yes' if sample_weights is not None else 'No'}")
        
        # XGBoost 파라미터 (Test 과적합 방지)
        xgb_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.03,  # 더 보수적
            'n_estimators': 2000,
            'max_depth': 6,  # 복잡도 감소
            'min_child_weight': 5,  # 더 보수적
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 2.0,  # 정규화 강화
            'reg_alpha': 0.1,
            'tree_method': 'hist',
            'random_state': 2025,
            'verbosity': 0
        }
        
        if self.config.get('use_gpu', False):
            xgb_params.update({'device': 'cuda'})
        
        # CV 학습
        cv_scores = []
        cv_predictions = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_strategy.split(X_train)):
            print(f"  폴드 {fold_idx + 1}/{self.cv_strategy.get_n_splits()} 학습 중...")
            
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 가중치 적용
            fold_weights = sample_weights[train_idx] if sample_weights is not None else None
            
            # 모델 학습
            model = XGBRegressor(**xgb_params)
            model.fit(
                X_fold_train, y_fold_train,
                sample_weight=fold_weights,
                eval_set=[(X_fold_val, y_fold_val)],
                early_stopping_rounds=100,
                verbose=False
            )
            
            # 검증 예측
            val_pred_log = model.predict(X_fold_val)
            val_pred = np.expm1(val_pred_log)
            val_true = np.expm1(y_fold_val)
            
            fold_mae = mean_absolute_error(val_true, val_pred)
            cv_scores.append(fold_mae)
            
            # Test 예측
            test_pred_log = model.predict(X_test)
            test_pred = np.expm1(test_pred_log)
            cv_predictions.append(test_pred)
        
        # 앙상블 예측
        final_predictions = np.mean(cv_predictions, axis=0)
        
        training_results = {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'feature_count': len(feature_cols),
            'features_used': feature_cols
        }
        
        print(f"\n🎯 학습 결과:")
        print(f"  CV MAE: {training_results['cv_mean']:.3f} ± {training_results['cv_std']:.3f}")
        print(f"  사용된 피처: {training_results['feature_count']}개")
        
        self.results['training'] = training_results
        
        return final_predictions, training_results
    
    def post_process_predictions(self, predictions: np.ndarray, 
                               train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """7단계: 예측값 후처리"""
        
        print("\n" + "="*60)
        print("7단계: 예측값 후처리")
        print("="*60)
        
        # 후처리기 생성
        processor = create_comprehensive_postprocessor(train)
        
        # 베이스라인 예측 (간단한 평균)
        baseline_predictions = None
        if all(col in test.columns for col in ['건물번호']):
            # 건물별 평균으로 베이스라인 생성
            building_means = train.groupby('건물번호')['전력소비량(kWh)'].mean()
            baseline_predictions = test['건물번호'].map(building_means).fillna(
                train['전력소비량(kWh)'].mean()
            ).values
        
        # 후처리 적용
        processed_predictions, processing_log = apply_post_processing_pipeline(
            predictions, test, processor,
            baseline_predictions=baseline_predictions,
            processing_steps=['clip_extreme', 'building_constraints', 'temporal_constraints', 'anomaly_detection']
        )
        
        print(f"\n🔧 후처리 결과:")
        summary = processing_log['summary']
        print(f"  수정된 예측값: {summary['changed_predictions']:,}개 ({summary['change_rate']*100:.1f}%)")
        print(f"  평균 변화량: {summary['avg_absolute_change']:.3f}")
        
        self.results['post_processing'] = processing_log
        
        return processed_predictions
    
    def save_results(self, predictions: np.ndarray, test: pd.DataFrame,
                    sample_submission_path: Optional[str] = None):
        """8단계: 결과 저장"""
        
        print("\n" + "="*60)
        print("8단계: 결과 저장")
        print("="*60)
        
        # 예측 결과 저장
        test_with_pred = test.copy()
        test_with_pred['prediction'] = predictions
        
        test_with_pred.to_csv(self.save_dir / "test_predictions.csv", 
                             index=False, encoding='utf-8-sig')
        
        # Sample submission 포맷으로 저장
        if sample_submission_path and Path(sample_submission_path).exists():
            sample_sub = pd.read_csv(sample_submission_path, encoding='utf-8-sig')
            
            # ID 매칭 (간단히 순서대로)
            submission = sample_sub.copy()
            submission.iloc[:, 1] = predictions[:len(submission)]
            
            submission.to_csv(self.save_dir / "submission.csv", 
                            index=False, encoding='utf-8-sig')
            print(f"  Sample submission 저장: submission.csv")
        
        # 파이프라인 결과 요약 저장
        import json
        with open(self.save_dir / "pipeline_results.json", 'w', encoding='utf-8') as f:
            # numpy 배열을 리스트로 변환
            results_serializable = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    results_serializable[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            results_serializable[key][k] = v.tolist()
                        elif isinstance(v, (np.int64, np.float64)):
                            results_serializable[key][k] = float(v)
                        else:
                            results_serializable[key][k] = v
                else:
                    results_serializable[key] = value
            
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"  결과 저장 완료: {self.save_dir}")
        print(f"    - test_predictions.csv: 상세 예측 결과")
        print(f"    - pipeline_results.json: 파이프라인 요약")
    
    def generate_summary_report(self):
        """최종 요약 리포트 생성"""
        
        print("\n" + "="*60)
        print("최종 요약 리포트")
        print("="*60)
        
        report = ["=" * 80]
        report.append("Test 과적합 최적화 파이프라인 실행 결과")
        report.append("=" * 80)
        report.append("")
        
        # 데이터 분석
        if 'data_analysis' in self.results:
            data_stats = self.results['data_analysis']
            report.append("1. 데이터 분석:")
            report.append(f"   - Train: {data_stats['train_shape'][0]:,}행 × {data_stats['train_shape'][1]}열")
            report.append(f"   - Test: {data_stats['test_shape'][0]:,}행 × {data_stats['test_shape'][1]}열")
            report.append(f"   - 유의한 분포 차이: {len(data_stats['significant_shifts'])}개 변수")
            report.append("")
        
        # 필터링
        if 'filtering' in self.results:
            filter_stats = self.results['filtering']
            report.append("2. Train 필터링:")
            report.append(f"   - 제거율: {filter_stats['removal_rate']:.1f}%")
            report.append(f"   - 최종 데이터: {filter_stats['final_count']:,}행")
            report.append("")
        
        # 피처 엔지니어링
        if 'feature_engineering' in self.results:
            feat_stats = self.results['feature_engineering']
            report.append("3. 피처 엔지니어링:")
            report.append(f"   - 추가된 피처: {feat_stats['total_new_features']}개")
            report.append(f"   - 최종 피처: {feat_stats['final_features']}개")
            report.append("")
        
        # 모델 학습
        if 'training' in self.results:
            train_stats = self.results['training']
            report.append("4. 모델 학습:")
            report.append(f"   - CV MAE: {train_stats['cv_mean']:.3f} ± {train_stats['cv_std']:.3f}")
            report.append(f"   - 사용된 피처: {train_stats['feature_count']}개")
            report.append("")
        
        # 후처리
        if 'post_processing' in self.results:
            post_stats = self.results['post_processing']
            summary = post_stats['summary']
            report.append("5. 후처리:")
            report.append(f"   - 수정된 예측값: {summary['changed_predictions']:,}개 ({summary['change_rate']*100:.1f}%)")
            report.append(f"   - 평균 변화량: {summary['avg_absolute_change']:.3f}")
            report.append("")
        
        # 예상 성능 개선
        report.append("6. 예상 성능 개선:")
        report.append("   - Train 필터링으로 분포 차이 감소")
        report.append("   - Importance sampling으로 Test 유사 샘플 강조")
        report.append("   - Test 적응형 피처로 계절적 특성 반영")
        report.append("   - Test 유사 CV로 검증 신뢰성 향상")
        report.append("   - 후처리로 극단값 및 이상치 제거")
        report.append("")
        
        report.append("7. 권장사항:")
        report.append("   - 본 파이프라인은 Test 성능 최적화에 특화됨")
        report.append("   - 대회 환경에서 Train-Test gap 최소화에 효과적")
        report.append("   - 실무에서는 일반화 성능도 함께 고려 필요")
        
        # 리포트 저장
        with open(self.save_dir / "final_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # 콘솔 출력
        for line in report:
            print(line)
        
        print(f"\n✅ 전체 파이프라인 완료!")
        print(f"   결과는 {self.save_dir}/ 디렉토리에 저장되었습니다.")

def create_default_config() -> Dict:
    """기본 설정 생성"""
    return {
        'filter_strategy': 'moderate',  # conservative, moderate, aggressive
        'cv_strategy': 'test_similarity',  # test_similarity, temperature_based, season_based, clustering_based
        'cv_splits': 5,
        'use_gpu': False,
        'save_dir': 'results'
    }

def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description="Test 과적합 최적화 통합 파이프라인")
    parser.add_argument("--train-path", type=str, 
                       default="../method_07/train_building_merged.csv",
                       help="Train 데이터 경로")
    parser.add_argument("--test-path", type=str,
                       default="../method_07/test_building_merged.csv", 
                       help="Test 데이터 경로")
    parser.add_argument("--sample-submission", type=str, default=None,
                       help="Sample submission 파일 경로")
    parser.add_argument("--filter-strategy", type=str, default="moderate",
                       choices=["conservative", "moderate", "aggressive"],
                       help="Train 필터링 강도")
    parser.add_argument("--cv-strategy", type=str, default="test_similarity",
                       choices=["test_similarity", "temperature_based", "season_based", "clustering_based"],
                       help="CV 전략")
    parser.add_argument("--cv-splits", type=int, default=5,
                       help="CV 폴드 수")
    parser.add_argument("--use-gpu", action="store_true",
                       help="GPU 사용 여부")
    parser.add_argument("--save-dir", type=str, default="results",
                       help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    # 설정 구성
    config = {
        'filter_strategy': args.filter_strategy,
        'cv_strategy': args.cv_strategy,
        'cv_splits': args.cv_splits,
        'use_gpu': args.use_gpu,
        'save_dir': args.save_dir
    }
    
    print("🎯 Test 과적합 최적화 파이프라인 시작")
    print(f"Train: {args.train_path}")
    print(f"Test: {args.test_path}")
    print(f"설정: {config}")
    
    # 파이프라인 실행
    pipeline = TestOverfittingPipeline(config)
    
    try:
        # 1. 데이터 로드 및 분석
        train, test = pipeline.load_and_analyze_data(args.train_path, args.test_path)
        
        # 2. Train 필터링
        filtered_train = pipeline.filter_train_data(train, test)
        
        # 3. Importance Sampling
        sample_weights = pipeline.calculate_sample_weights(filtered_train, test)
        
        # 4. 피처 엔지니어링
        enhanced_train, enhanced_test = pipeline.engineer_features(filtered_train, test)
        
        # 5. CV 전략 설정
        pipeline.setup_cv_strategy(enhanced_train, enhanced_test)
        
        # 6. 모델 학습
        predictions, training_results = pipeline.train_model(
            enhanced_train, enhanced_test, sample_weights
        )
        
        # 7. 후처리
        final_predictions = pipeline.post_process_predictions(
            predictions, enhanced_train, enhanced_test
        )
        
        # 8. 결과 저장
        pipeline.save_results(final_predictions, enhanced_test, args.sample_submission)
        
        # 9. 최종 리포트
        pipeline.generate_summary_report()
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())