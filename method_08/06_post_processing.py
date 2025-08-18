"""
Post-Processing Pipeline: 예측값 후처리로 성능 개선

모델 예측 후 다양한 후처리 기법을 적용하여 SMAPE 성능을 개선합니다.
특히 극단값 제거, 분포 조정, 합리성 검증 등을 통해 안정적인 예측을 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PostProcessor:
    """예측값 후처리 파이프라인"""
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Args:
            reference_data: 참조 데이터 (분포 조정용)
        """
        self.reference_data = reference_data
        self.fitted_stats = {}
        
    def fit_reference_statistics(self, reference_data: pd.DataFrame, 
                                target_col: str = '전력소비량(kWh)'):
        """참조 데이터 통계 학습"""
        
        self.reference_data = reference_data
        target_values = reference_data[target_col].dropna()
        
        self.fitted_stats = {
            'target_stats': {
                'mean': target_values.mean(),
                'std': target_values.std(),
                'median': target_values.median(),
                'q25': target_values.quantile(0.25),
                'q75': target_values.quantile(0.75),
                'q95': target_values.quantile(0.95),
                'q99': target_values.quantile(0.99),
                'min': target_values.min(),
                'max': target_values.max()
            }
        }
        
        # 건물별 통계
        if '건물번호' in reference_data.columns:
            building_stats = reference_data.groupby('건물번호')[target_col].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            self.fitted_stats['building_stats'] = building_stats.set_index('건물번호').to_dict('index')
        
        # 시간별 통계
        if '일시' in reference_data.columns:
            ref_with_time = reference_data.copy()
            ref_with_time['datetime'] = pd.to_datetime(ref_with_time['일시'], format='%Y%m%d %H', errors='coerce')
            ref_with_time['hour'] = ref_with_time['datetime'].dt.hour
            ref_with_time['weekday'] = ref_with_time['datetime'].dt.weekday
            
            hour_stats = ref_with_time.groupby('hour')[target_col].agg(['mean', 'std']).reset_index()
            weekday_stats = ref_with_time.groupby('weekday')[target_col].agg(['mean', 'std']).reset_index()
            
            self.fitted_stats['hour_stats'] = hour_stats.set_index('hour').to_dict('index')
            self.fitted_stats['weekday_stats'] = weekday_stats.set_index('weekday').to_dict('index')
        
        return self
    
    def clip_extreme_values(self, predictions: np.ndarray, 
                           method: str = 'percentile',
                           lower_bound: Optional[float] = None,
                           upper_bound: Optional[float] = None,
                           percentile_range: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
        """극단값 클리핑"""
        
        pred_clipped = predictions.copy()
        
        if method == 'percentile':
            if self.fitted_stats and 'target_stats' in self.fitted_stats:
                # 참조 데이터 기준 percentile
                lower = self.fitted_stats['target_stats']['q25'] * 0.1  # 매우 보수적 하한
                upper = self.fitted_stats['target_stats']['q99']  # 99% percentile 상한
            else:
                # 예측값 자체 percentile
                lower = np.percentile(predictions, percentile_range[0])
                upper = np.percentile(predictions, percentile_range[1])
                
        elif method == 'zscore':
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            lower = max(0, pred_mean - 3 * pred_std)
            upper = pred_mean + 3 * pred_std
            
        elif method == 'iqr':
            q25 = np.percentile(predictions, 25)
            q75 = np.percentile(predictions, 75)
            iqr = q75 - q25
            lower = max(0, q25 - 1.5 * iqr)
            upper = q75 + 1.5 * iqr
            
        elif method == 'manual':
            lower = lower_bound if lower_bound is not None else 0
            upper = upper_bound if upper_bound is not None else np.inf
            
        else:
            raise ValueError(f"지원하지 않는 클리핑 방법: {method}")
        
        pred_clipped = np.clip(pred_clipped, lower, upper)
        
        return pred_clipped
    
    def smooth_predictions(self, predictions: np.ndarray, 
                          data: pd.DataFrame,
                          method: str = 'moving_average',
                          window_size: int = 3) -> np.ndarray:
        """예측값 스무딩"""
        
        if method == 'moving_average':
            # 건물별 시간순 이동평균
            pred_smoothed = predictions.copy()
            
            if '건물번호' in data.columns and '일시' in data.columns:
                df_pred = data.copy()
                df_pred['prediction'] = predictions
                df_pred['datetime'] = pd.to_datetime(df_pred['일시'], format='%Y%m%d %H', errors='coerce')
                df_pred = df_pred.sort_values(['건물번호', 'datetime'])
                
                # 건물별 이동평균
                df_pred['pred_smoothed'] = df_pred.groupby('건물번호')['prediction'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean().reset_index(0, drop=True)
                
                # 원래 순서로 복원
                pred_smoothed = df_pred.loc[data.index, 'pred_smoothed'].values
                
        elif method == 'exponential_smoothing':
            # 지수 스무딩
            alpha = 0.3
            pred_smoothed = predictions.copy()
            
            if len(predictions) > 1:
                for i in range(1, len(predictions)):
                    pred_smoothed[i] = alpha * predictions[i] + (1 - alpha) * pred_smoothed[i-1]
                    
        else:
            pred_smoothed = predictions.copy()
        
        return pred_smoothed
    
    def apply_building_constraints(self, predictions: np.ndarray,
                                  data: pd.DataFrame) -> np.ndarray:
        """건물별 제약 조건 적용"""
        
        pred_constrained = predictions.copy()
        
        if '건물번호' not in data.columns:
            return pred_constrained
        
        for building_id in data['건물번호'].unique():
            building_mask = data['건물번호'] == building_id
            building_preds = pred_constrained[building_mask]
            
            # 건물별 참조 통계가 있는 경우
            if (self.fitted_stats and 'building_stats' in self.fitted_stats and 
                building_id in self.fitted_stats['building_stats']):
                
                building_stats = self.fitted_stats['building_stats'][building_id]
                
                # 건물별 합리적 범위 제한
                building_min = max(0, building_stats['min'] * 0.5)  # 최소값의 50%
                building_max = building_stats['max'] * 1.5  # 최대값의 150%
                
                building_preds = np.clip(building_preds, building_min, building_max)
                
                # 건물별 평균 대비 극단값 제한
                building_mean = building_stats['mean']
                building_std = building_stats.get('std', building_mean * 0.5)
                
                # 평균의 5배를 넘지 않도록
                upper_limit = building_mean * 5
                building_preds = np.minimum(building_preds, upper_limit)
                
                pred_constrained[building_mask] = building_preds
        
        return pred_constrained
    
    def apply_temporal_constraints(self, predictions: np.ndarray,
                                  data: pd.DataFrame) -> np.ndarray:
        """시간적 제약 조건 적용"""
        
        pred_constrained = predictions.copy()
        
        if '일시' not in data.columns:
            return pred_constrained
        
        # 시간 정보 추출
        df_pred = data.copy()
        df_pred['prediction'] = pred_constrained
        df_pred['datetime'] = pd.to_datetime(df_pred['일시'], format='%Y%m%d %H', errors='coerce')
        df_pred['hour'] = df_pred['datetime'].dt.hour
        df_pred['weekday'] = df_pred['datetime'].dt.weekday
        
        # 시간대별 제약
        if self.fitted_stats and 'hour_stats' in self.fitted_stats:
            for hour in df_pred['hour'].unique():
                if hour in self.fitted_stats['hour_stats']:
                    hour_mask = df_pred['hour'] == hour
                    hour_stats = self.fitted_stats['hour_stats'][hour]
                    
                    hour_mean = hour_stats['mean']
                    hour_std = hour_stats.get('std', hour_mean * 0.5)
                    
                    # 시간대 평균의 3배를 넘지 않도록
                    hour_upper = hour_mean * 3
                    df_pred.loc[hour_mask, 'prediction'] = np.minimum(
                        df_pred.loc[hour_mask, 'prediction'], hour_upper
                    )
        
        # 야간 시간대 특별 제약 (에너지 절약)
        night_hours = [0, 1, 2, 3, 4, 5]
        night_mask = df_pred['hour'].isin(night_hours)
        
        if night_mask.any():
            # 야간에는 일반적으로 소비량이 낮음
            day_median = df_pred[~night_mask]['prediction'].median()
            night_upper = day_median * 0.7  # 주간 중앙값의 70%
            
            df_pred.loc[night_mask, 'prediction'] = np.minimum(
                df_pred.loc[night_mask, 'prediction'], night_upper
            )
        
        return df_pred['prediction'].values
    
    def detect_and_fix_anomalies(self, predictions: np.ndarray,
                                data: pd.DataFrame,
                                method: str = 'isolation_forest') -> np.ndarray:
        """이상치 탐지 및 수정"""
        
        pred_fixed = predictions.copy()
        
        if method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                
                # 예측값과 몇 가지 컨텍스트 피처로 이상치 탐지
                features = []
                features.append(predictions.reshape(-1, 1))
                
                if '건물번호' in data.columns:
                    building_encoded = pd.get_dummies(data['건물번호']).values
                    features.append(building_encoded)
                
                if '일시' in data.columns:
                    data_temp = data.copy()
                    data_temp['datetime'] = pd.to_datetime(data_temp['일시'], format='%Y%m%d %H', errors='coerce')
                    data_temp['hour'] = data_temp['datetime'].dt.hour
                    data_temp['weekday'] = data_temp['datetime'].dt.weekday
                    features.append(data_temp[['hour', 'weekday']].values)
                
                X = np.hstack(features) if len(features) > 1 else features[0]
                
                # Isolation Forest 적용
                iso_forest = IsolationForest(contamination=0.05, random_state=2025)
                outlier_labels = iso_forest.fit_predict(X)
                
                # 이상치 수정 (중앙값으로 대체)
                outlier_mask = outlier_labels == -1
                if outlier_mask.any():
                    pred_fixed[outlier_mask] = np.median(predictions[~outlier_mask])
                    
            except ImportError:
                # sklearn이 없으면 통계적 방법 사용
                method = 'statistical'
        
        if method == 'statistical':
            # Z-score 기반 이상치 탐지
            z_scores = np.abs(stats.zscore(predictions))
            outlier_mask = z_scores > 3
            
            if outlier_mask.any():
                pred_fixed[outlier_mask] = np.median(predictions[~outlier_mask])
        
        return pred_fixed
    
    def apply_ensemble_correction(self, predictions: np.ndarray,
                                 baseline_predictions: Optional[np.ndarray] = None,
                                 correction_weight: float = 0.1) -> np.ndarray:
        """베이스라인과의 앙상블 보정"""
        
        if baseline_predictions is None:
            return predictions
        
        # 예측값과 베이스라인의 가중 평균
        corrected = (1 - correction_weight) * predictions + correction_weight * baseline_predictions
        
        return corrected
    
    def apply_distribution_adjustment(self, predictions: np.ndarray,
                                    target_distribution: Optional[Dict] = None) -> np.ndarray:
        """분포 조정 (매우 조심스럽게 적용)"""
        
        if target_distribution is None or not self.fitted_stats:
            return predictions
        
        # 현재 예측 분포
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # 참조 분포
        if 'target_stats' in self.fitted_stats:
            ref_mean = self.fitted_stats['target_stats']['mean']
            ref_std = self.fitted_stats['target_stats']['std']
            
            # 분포 차이가 클 때만 조정 (과도한 조정 방지)
            mean_ratio = ref_mean / pred_mean if pred_mean > 0 else 1.0
            std_ratio = ref_std / pred_std if pred_std > 0 else 1.0
            
            # 보수적 조정 (최대 20% 변화)
            mean_ratio = np.clip(mean_ratio, 0.8, 1.2)
            std_ratio = np.clip(std_ratio, 0.8, 1.2)
            
            # 분포 조정 적용
            adjusted = predictions * mean_ratio
            adjusted = (adjusted - np.mean(adjusted)) * std_ratio + np.mean(adjusted)
            
            return np.maximum(adjusted, 0)  # 음수 방지
        
        return predictions

def create_comprehensive_postprocessor(train_data: pd.DataFrame,
                                     target_col: str = '전력소비량(kWh)') -> PostProcessor:
    """종합적인 후처리기 생성"""
    
    processor = PostProcessor()
    processor.fit_reference_statistics(train_data, target_col)
    
    return processor

def apply_post_processing_pipeline(predictions: np.ndarray,
                                 data: pd.DataFrame,
                                 processor: PostProcessor,
                                 baseline_predictions: Optional[np.ndarray] = None,
                                 processing_steps: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
    """후처리 파이프라인 적용"""
    
    if processing_steps is None:
        processing_steps = [
            'clip_extreme',
            'building_constraints', 
            'temporal_constraints',
            'anomaly_detection',
            'smooth',
            'ensemble_correction'
        ]
    
    print("🔧 예측값 후처리 파이프라인 시작...")
    
    processed_predictions = predictions.copy()
    processing_log = {
        'original_stats': {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'negative_count': np.sum(predictions < 0)
        },
        'steps_applied': [],
        'step_stats': {}
    }
    
    original_predictions = predictions.copy()
    step_predictions = processed_predictions.copy()
    
    for step in processing_steps:
        print(f"   적용 중: {step}...")
        prev_predictions = step_predictions.copy()
        
        if step == 'clip_extreme':
            step_predictions = processor.clip_extreme_values(
                step_predictions, method='percentile'
            )
            
        elif step == 'building_constraints':
            step_predictions = processor.apply_building_constraints(
                step_predictions, data
            )
            
        elif step == 'temporal_constraints':
            step_predictions = processor.apply_temporal_constraints(
                step_predictions, data
            )
            
        elif step == 'anomaly_detection':
            step_predictions = processor.detect_and_fix_anomalies(
                step_predictions, data, method='statistical'
            )
            
        elif step == 'smooth':
            step_predictions = processor.smooth_predictions(
                step_predictions, data, method='moving_average', window_size=3
            )
            
        elif step == 'ensemble_correction':
            if baseline_predictions is not None:
                step_predictions = processor.apply_ensemble_correction(
                    step_predictions, baseline_predictions, correction_weight=0.1
                )
        
        # 단계별 변화 기록
        changes = np.sum(step_predictions != prev_predictions)
        avg_change = np.mean(np.abs(step_predictions - prev_predictions))
        
        processing_log['steps_applied'].append(step)
        processing_log['step_stats'][step] = {
            'changed_predictions': changes,
            'avg_change': avg_change,
            'new_mean': np.mean(step_predictions),
            'new_std': np.std(step_predictions)
        }
    
    # 최종 통계
    processing_log['final_stats'] = {
        'mean': np.mean(step_predictions),
        'std': np.std(step_predictions),
        'min': np.min(step_predictions),
        'max': np.max(step_predictions),
        'negative_count': np.sum(step_predictions < 0)
    }
    
    # 전체 변화 요약
    total_changes = np.sum(step_predictions != original_predictions)
    avg_total_change = np.mean(np.abs(step_predictions - original_predictions))
    
    processing_log['summary'] = {
        'total_predictions': len(predictions),
        'changed_predictions': total_changes,
        'change_rate': total_changes / len(predictions),
        'avg_absolute_change': avg_total_change
    }
    
    print(f"✅ 후처리 완료: {total_changes:,}개 예측값 수정 ({total_changes/len(predictions)*100:.1f}%)")
    
    return step_predictions, processing_log

def analyze_post_processing_impact(original_predictions: np.ndarray,
                                 processed_predictions: np.ndarray,
                                 true_values: Optional[np.ndarray] = None,
                                 save_dir: str = "eda") -> Dict:
    """후처리 효과 분석"""
    
    Path(save_dir).mkdir(exist_ok=True)
    
    analysis = {
        'distribution_changes': {},
        'extreme_value_changes': {},
        'performance_changes': {}
    }
    
    # 1. 분포 변화 분석
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 히스토그램 비교
    axes[0,0].hist(original_predictions, bins=50, alpha=0.7, label='Original', density=True)
    axes[0,0].hist(processed_predictions, bins=50, alpha=0.7, label='Processed', density=True)
    axes[0,0].set_title('예측값 분포 비교')
    axes[0,0].legend()
    
    # 산점도 비교
    axes[0,1].scatter(original_predictions, processed_predictions, alpha=0.5, s=1)
    axes[0,1].plot([original_predictions.min(), original_predictions.max()], 
                   [original_predictions.min(), original_predictions.max()], 'r--', label='y=x')
    axes[0,1].set_xlabel('Original Predictions')
    axes[0,1].set_ylabel('Processed Predictions')
    axes[0,1].set_title('예측값 변화 산점도')
    axes[0,1].legend()
    
    # 차이 분포
    differences = processed_predictions - original_predictions
    axes[1,0].hist(differences, bins=50, alpha=0.7)
    axes[1,0].set_title('예측값 변화량 분포')
    axes[1,0].set_xlabel('Processed - Original')
    
    # 극단값 비교
    original_q99 = np.percentile(original_predictions, 99)
    processed_q99 = np.percentile(processed_predictions, 99)
    
    extreme_mask = original_predictions > original_q99
    
    if extreme_mask.any():
        axes[1,1].scatter(original_predictions[extreme_mask], 
                         processed_predictions[extreme_mask], alpha=0.7, label='Extreme Values')
        axes[1,1].plot([original_predictions.min(), original_predictions.max()], 
                       [original_predictions.min(), original_predictions.max()], 'r--', label='y=x')
        axes[1,1].set_xlabel('Original Predictions')
        axes[1,1].set_ylabel('Processed Predictions')
        axes[1,1].set_title('극단값 변화')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/post_processing_impact_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 통계적 변화 분석
    analysis['distribution_changes'] = {
        'mean_change': np.mean(processed_predictions) - np.mean(original_predictions),
        'std_change': np.std(processed_predictions) - np.std(original_predictions),
        'median_change': np.median(processed_predictions) - np.median(original_predictions),
        'q99_change': processed_q99 - original_q99
    }
    
    analysis['extreme_value_changes'] = {
        'original_extreme_count': np.sum(original_predictions > original_q99),
        'processed_extreme_count': np.sum(processed_predictions > original_q99),
        'negative_removed': np.sum(original_predictions < 0) - np.sum(processed_predictions < 0)
    }
    
    # 3. 성능 변화 (true_values가 있는 경우)
    if true_values is not None:
        def smape(y_true, y_pred):
            return 200.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        
        original_smape = smape(true_values, original_predictions)
        processed_smape = smape(true_values, processed_predictions)
        
        analysis['performance_changes'] = {
            'original_smape': original_smape,
            'processed_smape': processed_smape,
            'smape_improvement': original_smape - processed_smape
        }
    
    return analysis

def save_post_processing_report(processing_log: Dict, analysis: Dict,
                              save_path: str = "eda/post_processing_report.txt"):
    """후처리 분석 리포트 저장"""
    
    report = ["=" * 80]
    report.append("예측값 후처리 분석 리포트")
    report.append("=" * 80)
    report.append("")
    
    # 처리 요약
    report.append("처리 요약:")
    report.append("-" * 40)
    summary = processing_log['summary']
    report.append(f"전체 예측값: {summary['total_predictions']:,}개")
    report.append(f"수정된 예측값: {summary['changed_predictions']:,}개 ({summary['change_rate']*100:.1f}%)")
    report.append(f"평균 절대 변화량: {summary['avg_absolute_change']:.3f}")
    report.append("")
    
    # 단계별 상세
    report.append("단계별 변화:")
    report.append("-" * 40)
    for step in processing_log['steps_applied']:
        step_stats = processing_log['step_stats'][step]
        report.append(f"\n{step.upper()}:")
        report.append(f"  - 수정된 예측값: {step_stats['changed_predictions']:,}개")
        report.append(f"  - 평균 변화량: {step_stats['avg_change']:.3f}")
        report.append(f"  - 새 평균: {step_stats['new_mean']:.3f}")
    
    # 분포 변화
    if 'distribution_changes' in analysis:
        dist_changes = analysis['distribution_changes']
        report.append("\n분포 변화:")
        report.append("-" * 40)
        report.append(f"평균 변화: {dist_changes['mean_change']:.3f}")
        report.append(f"표준편차 변화: {dist_changes['std_change']:.3f}")
        report.append(f"중앙값 변화: {dist_changes['median_change']:.3f}")
        report.append(f"99% 분위수 변화: {dist_changes['q99_change']:.3f}")
    
    # 극단값 변화
    if 'extreme_value_changes' in analysis:
        extreme_changes = analysis['extreme_value_changes']
        report.append("\n극단값 변화:")
        report.append("-" * 40)
        report.append(f"원본 극단값 개수: {extreme_changes['original_extreme_count']:,}")
        report.append(f"처리 후 극단값 개수: {extreme_changes['processed_extreme_count']:,}")
        report.append(f"제거된 음수 예측값: {extreme_changes['negative_removed']:,}")
    
    # 성능 변화
    if 'performance_changes' in analysis:
        perf_changes = analysis['performance_changes']
        report.append("\n성능 변화:")
        report.append("-" * 40)
        report.append(f"원본 SMAPE: {perf_changes['original_smape']:.3f}%")
        report.append(f"처리 후 SMAPE: {perf_changes['processed_smape']:.3f}%")
        report.append(f"SMAPE 개선: {perf_changes['smape_improvement']:.3f}%")
    
    # 권장사항
    report.append("\n권장사항:")
    report.append("-" * 40)
    report.append("1. 극단값 클리핑은 SMAPE 성능 개선에 효과적")
    report.append("2. 건물별/시간대별 제약은 예측의 합리성 향상")
    report.append("3. 과도한 스무딩은 피하고 보수적 후처리 권장")
    report.append("4. 베이스라인과의 앙상블은 안정성 향상에 도움")
    
    Path(save_path).parent.mkdir(exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def main():
    """메인 실행 함수"""
    
    print("🔧 Post-Processing 파이프라인 테스트...")
    
    # 데이터 로드
    train = pd.read_csv("../method_07/train_building_merged.csv", encoding='utf-8-sig')
    
    print(f"Train 데이터: {len(train):,}행")
    
    # EDA 디렉토리 생성
    eda_dir = "eda"
    Path(eda_dir).mkdir(exist_ok=True)
    
    # 샘플 예측값 생성 (테스트용)
    np.random.seed(2025)
    target_values = train['전력소비량(kWh)'].dropna().values
    
    # 노이즈가 있는 예측값 시뮬레이션
    sample_predictions = target_values * (1 + np.random.normal(0, 0.2, len(target_values)))
    
    # 일부 극단값 추가
    extreme_indices = np.random.choice(len(sample_predictions), size=int(len(sample_predictions) * 0.05), replace=False)
    sample_predictions[extreme_indices] *= np.random.uniform(3, 10, len(extreme_indices))
    
    # 일부 음수값 추가
    negative_indices = np.random.choice(len(sample_predictions), size=int(len(sample_predictions) * 0.02), replace=False)
    sample_predictions[negative_indices] *= -1
    
    print(f"시뮬레이션 예측값 생성: {len(sample_predictions):,}개")
    print(f"   - 음수 예측값: {np.sum(sample_predictions < 0)}개")
    print(f"   - 극단값 (원본 99%): {np.sum(sample_predictions > np.percentile(target_values, 99))}개")
    
    # 후처리기 생성
    processor = create_comprehensive_postprocessor(train.dropna(subset=['전력소비량(kWh)']))
    
    # 베이스라인 예측값 (건물×시간×요일 평균)
    baseline_predictions = None
    if all(col in train.columns for col in ['건물번호', '일시']):
        train_sample = train.dropna(subset=['전력소비량(kWh)'])
        
        # 간단한 베이스라인 생성
        train_sample['datetime'] = pd.to_datetime(train_sample['일시'], format='%Y%m%d %H', errors='coerce')
        train_sample['hour'] = train_sample['datetime'].dt.hour
        train_sample['weekday'] = train_sample['datetime'].dt.weekday
        
        baseline_stats = train_sample.groupby(['건물번호', 'hour', 'weekday'])['전력소비량(kWh)'].mean()
        
        # 샘플 데이터에 베이스라인 매핑
        sample_data = train.dropna(subset=['전력소비량(kWh)']).copy()
        sample_data['datetime'] = pd.to_datetime(sample_data['일시'], format='%Y%m%d %H', errors='coerce')
        sample_data['hour'] = sample_data['datetime'].dt.hour
        sample_data['weekday'] = sample_data['datetime'].dt.weekday
        
        baseline_predictions = []
        for _, row in sample_data.iterrows():
            key = (row['건물번호'], row['hour'], row['weekday'])
            baseline_val = baseline_stats.get(key, sample_data['전력소비량(kWh)'].mean())
            baseline_predictions.append(baseline_val)
        
        baseline_predictions = np.array(baseline_predictions)
    
    # 후처리 적용
    processed_predictions, processing_log = apply_post_processing_pipeline(
        sample_predictions,
        train.dropna(subset=['전력소비량(kWh)']),
        processor,
        baseline_predictions=baseline_predictions
    )
    
    # 효과 분석
    print("\n📊 후처리 효과 분석...")
    analysis = analyze_post_processing_impact(
        sample_predictions,
        processed_predictions,
        true_values=target_values,
        save_dir=eda_dir
    )
    
    # 리포트 저장
    save_post_processing_report(processing_log, analysis, f"{eda_dir}/post_processing_report.txt")
    
    # 결과 요약
    print(f"\n✅ Post-Processing 테스트 완료!")
    print(f"   원본 예측값: 평균 {np.mean(sample_predictions):.1f}, 표준편차 {np.std(sample_predictions):.1f}")
    print(f"   처리 후: 평균 {np.mean(processed_predictions):.1f}, 표준편차 {np.std(processed_predictions):.1f}")
    
    if 'performance_changes' in analysis:
        perf = analysis['performance_changes']
        print(f"   SMAPE 개선: {perf['original_smape']:.3f}% → {perf['processed_smape']:.3f}% ({perf['smape_improvement']:.3f}% 개선)")
    
    print(f"   결과는 {eda_dir}/ 디렉토리에 저장되었습니다.")
    
    return processor, processing_log, analysis

if __name__ == "__main__":
    processor, processing_log, analysis = main()