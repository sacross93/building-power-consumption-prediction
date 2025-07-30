"""
Model Performance Comparison: Original vs Improved Preprocessing
===============================================================

기존 전처리와 개선된 전처리의 모델 성능 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, engineer_features, smape
from ts_validation import TimeSeriesCV
from improved_preprocessing import ImprovedPreprocessor


class ModelComparison:
    """모델 성능 비교 클래스."""
    
    def __init__(self):
        self.results = {}
        
    def prepare_original_data(self, train_df, test_df):
        """기존 전처리 방식으로 데이터 준비."""
        print("Preparing data with original preprocessing...")
        
        # 기존 engineer_features 사용
        train_fe, test_fe = engineer_features(train_df.copy(), test_df.copy())
        
        # 피처 준비
        drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
        feature_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]
        
        X_train = train_fe[feature_cols].copy()
        X_test = test_fe[feature_cols].copy()
        y_train = train_fe['전력소비량(kWh)']
        
        # 카테고리 인코딩
        categorical_cols = ['건물번호', 'building_type']
        encoders = {}
        
        for col in categorical_cols:
            if col in X_train.columns:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = X_test[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                encoders[col] = le
        
        # 객체 타입 처리
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = X_test[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                ).fillna(-1)
                encoders[col] = le
            elif not pd.api.types.is_numeric_dtype(X_train[col]):
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        
        return X_train, X_test, y_train, train_fe
    
    def create_base_model(self):
        """기본 XGBoost 모델 생성."""
        return xgb.XGBRegressor(
            max_depth=8,
            n_estimators=400,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0
        )
    
    def create_ensemble_model(self):
        """앙상블 모델 생성."""
        xgb_model = xgb.XGBRegressor(
            max_depth=8,
            n_estimators=400,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0
        )
        
        lgb_model = lgb.LGBMRegressor(
            max_depth=8,
            n_estimators=400,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1
        )
        
        return VotingRegressor([
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ])
    
    def evaluate_model(self, X, y, datetime_series, model, model_name, is_log_target=False):
        """시계열 교차검증으로 모델 평가."""
        print(f"Evaluating {model_name}...")
        
        ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
        temp_df = pd.DataFrame({'datetime': datetime_series})
        splits = ts_cv.split(temp_df, 'datetime')
        
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # 모델 훈련
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            # 로그 변환된 타겟의 경우 역변환
            if is_log_target:
                y_val_original = np.expm1(y_val_fold)
                y_pred_original = np.expm1(y_pred)
                fold_smape = smape(y_val_original.values, y_pred_original)
            else:
                fold_smape = smape(y_val_fold.values, y_pred)
            
            fold_scores.append(fold_smape)
            print(f"  Fold {fold + 1}: SMAPE = {fold_smape:.4f}")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"  {model_name} CV Score: {mean_score:.4f} (±{std_score:.4f})")
        
        return mean_score, std_score, fold_scores
    
    def run_comparison(self):
        """전체 비교 실행."""
        print("=" * 80)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # 데이터 로드
        data_dir = Path('../data')
        train_path = data_dir / 'train.csv'
        test_path = data_dir / 'test.csv'
        building_path = data_dir / 'building_info.csv'
        
        train_df, test_df = load_data(train_path, test_path, building_path)
        
        # 1. 기존 전처리 평가
        print("\n1. ORIGINAL PREPROCESSING EVALUATION")
        print("-" * 50)
        
        X_orig, X_test_orig, y_orig, train_orig = self.prepare_original_data(train_df, test_df)
        
        # 기존 모델들 평가
        base_model_orig = self.create_base_model()
        ensemble_model_orig = self.create_ensemble_model()
        
        orig_base_mean, orig_base_std, orig_base_folds = self.evaluate_model(
            X_orig, y_orig, train_orig['datetime'], base_model_orig, "Original XGBoost"
        )
        
        orig_ensemble_mean, orig_ensemble_std, orig_ensemble_folds = self.evaluate_model(
            X_orig, y_orig, train_orig['datetime'], ensemble_model_orig, "Original Ensemble"
        )
        
        # 2. 개선된 전처리 평가
        print("\n2. IMPROVED PREPROCESSING EVALUATION")
        print("-" * 50)
        
        preprocessor = ImprovedPreprocessor()
        X_improved, X_test_improved, y_improved = preprocessor.fit_transform(train_df, test_df)
        
        # 개선된 모델들 평가
        base_model_improved = self.create_base_model()
        ensemble_model_improved = self.create_ensemble_model()
        
        # datetime 재구성 (개선된 전처리에서는 datetime이 drop됨)
        train_processed, _ = engineer_features(train_df.copy(), test_df.copy())
        
        improved_base_mean, improved_base_std, improved_base_folds = self.evaluate_model(
            X_improved, y_improved, train_processed['datetime'], base_model_improved, 
            "Improved XGBoost", is_log_target=True
        )
        
        improved_ensemble_mean, improved_ensemble_std, improved_ensemble_folds = self.evaluate_model(
            X_improved, y_improved, train_processed['datetime'], ensemble_model_improved, 
            "Improved Ensemble", is_log_target=True
        )
        
        # 3. 결과 저장
        self.results = {
            'original': {
                'xgboost': {'mean': orig_base_mean, 'std': orig_base_std, 'folds': orig_base_folds},
                'ensemble': {'mean': orig_ensemble_mean, 'std': orig_ensemble_std, 'folds': orig_ensemble_folds}
            },
            'improved': {
                'xgboost': {'mean': improved_base_mean, 'std': improved_base_std, 'folds': improved_base_folds},
                'ensemble': {'mean': improved_ensemble_mean, 'std': improved_ensemble_std, 'folds': improved_ensemble_folds}
            }
        }
        
        # 4. 결과 시각화
        self.create_comparison_plots()
        
        # 5. 결과 리포트
        self.create_comparison_report()
        
        return self.results
    
    def create_comparison_plots(self):
        """비교 결과 시각화."""
        print("\nCreating comparison plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 평균 SMAPE 비교
        models = ['XGBoost', 'Ensemble']
        original_means = [self.results['original']['xgboost']['mean'], 
                         self.results['original']['ensemble']['mean']]
        improved_means = [self.results['improved']['xgboost']['mean'], 
                         self.results['improved']['ensemble']['mean']]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, original_means, width, label='Original', alpha=0.7, color='lightcoral')
        bars2 = ax1.bar(x + width/2, improved_means, width, label='Improved', alpha=0.7, color='lightblue')
        
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('SMAPE (%)')
        ax1.set_title('Mean SMAPE Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.3f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. 표준편차 비교
        original_stds = [self.results['original']['xgboost']['std'], 
                        self.results['original']['ensemble']['std']]
        improved_stds = [self.results['improved']['xgboost']['std'], 
                        self.results['improved']['ensemble']['std']]
        
        bars3 = ax2.bar(x - width/2, original_stds, width, label='Original', alpha=0.7, color='lightcoral')
        bars4 = ax2.bar(x + width/2, improved_stds, width, label='Improved', alpha=0.7, color='lightblue')
        
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('SMAPE Standard Deviation')
        ax2.set_title('SMAPE Stability Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Fold별 XGBoost 성능
        folds = [1, 2, 3]
        ax3.plot(folds, self.results['original']['xgboost']['folds'], 
                'o-', label='Original XGBoost', linewidth=2, markersize=8, color='red')
        ax3.plot(folds, self.results['improved']['xgboost']['folds'], 
                'o-', label='Improved XGBoost', linewidth=2, markersize=8, color='blue')
        
        ax3.set_xlabel('CV Fold')
        ax3.set_ylabel('SMAPE (%)')
        ax3.set_title('XGBoost Performance by Fold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(folds)
        
        # 4. Fold별 Ensemble 성능
        ax4.plot(folds, self.results['original']['ensemble']['folds'], 
                'o-', label='Original Ensemble', linewidth=2, markersize=8, color='red')
        ax4.plot(folds, self.results['improved']['ensemble']['folds'], 
                'o-', label='Improved Ensemble', linewidth=2, markersize=8, color='blue')
        
        ax4.set_xlabel('CV Fold')
        ax4.set_ylabel('SMAPE (%)')
        ax4.set_title('Ensemble Performance by Fold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(folds)
        
        plt.tight_layout()
        plt.savefig('./visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Comparison plots saved: visualizations/model_comparison.png")
    
    def create_comparison_report(self):
        """비교 결과 리포트 생성."""
        report_path = Path('./visualizations/model_comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Model Performance Comparison Report\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # 개선 효과 계산
            xgb_improvement = (
                (self.results['original']['xgboost']['mean'] - self.results['improved']['xgboost']['mean']) /
                self.results['original']['xgboost']['mean'] * 100
            )
            
            ensemble_improvement = (
                (self.results['original']['ensemble']['mean'] - self.results['improved']['ensemble']['mean']) /
                self.results['original']['ensemble']['mean'] * 100
            )
            
            f.write(f"- **XGBoost Improvement**: {xgb_improvement:.2f}%\n")
            f.write(f"- **Ensemble Improvement**: {ensemble_improvement:.2f}%\n\n")
            
            f.write("## Detailed Results\n\n")
            
            # 결과 테이블
            f.write("| Model | Preprocessing | Mean SMAPE | Std SMAPE | Best Fold | Worst Fold |\n")
            f.write("|-------|---------------|------------|-----------|-----------|------------|\n")
            
            for preprocessing in ['original', 'improved']:
                for model in ['xgboost', 'ensemble']:
                    result = self.results[preprocessing][model]
                    best_fold = min(result['folds'])
                    worst_fold = max(result['folds'])
                    
                    f.write(f"| {model.title()} | {preprocessing.title()} | "
                           f"{result['mean']:.4f} | {result['std']:.4f} | "
                           f"{best_fold:.4f} | {worst_fold:.4f} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            if xgb_improvement > 0:
                f.write(f"✅ **XGBoost showed {xgb_improvement:.2f}% improvement** with improved preprocessing\n")
            else:
                f.write(f"❌ **XGBoost showed {abs(xgb_improvement):.2f}% degradation** with improved preprocessing\n")
            
            if ensemble_improvement > 0:
                f.write(f"✅ **Ensemble showed {ensemble_improvement:.2f}% improvement** with improved preprocessing\n")
            else:
                f.write(f"❌ **Ensemble showed {abs(ensemble_improvement):.2f}% degradation** with improved preprocessing\n")
            
            f.write("\n## Preprocessing Impact Analysis\n\n")
            
            f.write("### Improved Preprocessing Benefits:\n")
            f.write("1. **Target Transformation**: Log transformation reduced skewness\n")
            f.write("2. **Feature Scaling**: RobustScaler normalized feature ranges\n")
            f.write("3. **Outlier Treatment**: IQR-based clipping improved robustness\n")
            f.write("4. **Multicollinearity Removal**: VIF analysis reduced overfitting risk\n")
            f.write("5. **Advanced Features**: Degree-days, heat index, building-specific features\n\n")
            
            f.write("### Potential Issues:\n")
            f.write("- Information loss from aggressive outlier treatment\n")
            f.write("- Over-preprocessing may remove useful signal\n")
            f.write("- Log transformation requires careful inverse transformation\n\n")
            
            f.write("## Recommendations\n\n")
            
            best_original = min(self.results['original']['xgboost']['mean'], 
                              self.results['original']['ensemble']['mean'])
            best_improved = min(self.results['improved']['xgboost']['mean'], 
                              self.results['improved']['ensemble']['mean'])
            
            if best_improved < best_original:
                f.write("🎯 **Recommendation**: Use improved preprocessing pipeline\n")
                f.write(f"- Best improved model SMAPE: {best_improved:.4f}\n")
                f.write(f"- Best original model SMAPE: {best_original:.4f}\n")
                f.write(f"- Overall improvement: {((best_original - best_improved) / best_original * 100):.2f}%\n")
            else:
                f.write("🎯 **Recommendation**: Stick with original preprocessing\n")
                f.write(f"- Best original model SMAPE: {best_original:.4f}\n")
                f.write(f"- Best improved model SMAPE: {best_improved:.4f}\n")
                f.write("- Consider selective adoption of specific improvements\n")
        
        print(f"📄 Comparison report saved: {report_path}")


if __name__ == "__main__":
    comparison = ModelComparison()
    results = comparison.run_comparison()
    
    print("\n🎯 Model comparison completed!")
    print("Check visualizations/model_comparison.png and model_comparison_report.md for detailed results.")