"""
Advanced Ensemble with CatBoost and Optimized Weights
=====================================================

3-model 앙상블 + 가중치 최적화 + Stacking 앙상블
11.8 SMAPE에서 추가 개선 목표
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, smape
from ts_validation import TimeSeriesCV
from improved_preprocessing import ImprovedPreprocessor


class AdvancedEnsemble:
    """고급 앙상블 클래스."""
    
    def __init__(self):
        self.base_models = {}
        self.ensemble_models = {}
        self.validation_scores = {}
        
    def create_optimized_base_models(self):
        """최적화된 기본 모델들 생성."""
        print("🔧 Creating optimized base models...")
        
        # XGBoost (기존 대비 강화된 파라미터)
        self.base_models['xgboost'] = xgb.XGBRegressor(
            max_depth=7,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            reg_alpha=1.5,
            reg_lambda=2.0,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            verbosity=0
        )
        
        # LightGBM (최적화된 파라미터)
        self.base_models['lightgbm'] = lgb.LGBMRegressor(
            max_depth=7,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.5,
            reg_lambda=2.0,
            min_child_weight=3,
            min_child_samples=20,
            num_leaves=100,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            verbosity=-1
        )
        
        # CatBoost (새로 추가)
        self.base_models['catboost'] = cb.CatBoostRegressor(
            depth=6,
            iterations=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bylevel=0.8,
            reg_lambda=2.0,
            min_data_in_leaf=20,
            max_leaves=100,
            bootstrap_type='Bayesian',
            bagging_temperature=1.0,
            random_seed=42,
            verbose=False
        )
        
        print(f"✅ Created {len(self.base_models)} optimized base models")
    
    def create_ensemble_models(self):
        """다양한 앙상블 모델들 생성."""
        print("🎭 Creating ensemble models...")
        
        # 1. 단순 평균 앙상블
        self.ensemble_models['simple_average'] = VotingRegressor([
            ('xgb', self.base_models['xgboost']),
            ('lgb', self.base_models['lightgbm']),
            ('cb', self.base_models['catboost'])
        ])
        
        # 2. 가중 평균 앙상블 (성능 기반 가중치)
        self.ensemble_models['weighted_average'] = VotingRegressor([
            ('xgb', self.base_models['xgboost']),
            ('lgb', self.base_models['lightgbm']),
            ('cb', self.base_models['catboost'])
        ], weights=[0.4, 0.35, 0.25])  # XGBoost 최고 가중치
        
        # 3. Stacking 앙상블 (Ridge 메타모델)
        self.ensemble_models['stacking_ridge'] = StackingRegressor(
            estimators=[
                ('xgb', self.base_models['xgboost']),
                ('lgb', self.base_models['lightgbm']),
                ('cb', self.base_models['catboost'])
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3  # 내부 교차검증
        )
        
        # 4. Stacking 앙상블 (ElasticNet 메타모델)
        self.ensemble_models['stacking_elastic'] = StackingRegressor(
            estimators=[
                ('xgb', self.base_models['xgboost']),
                ('lgb', self.base_models['lightgbm']),
                ('cb', self.base_models['catboost'])
            ],
            final_estimator=ElasticNet(alpha=0.1, l1_ratio=0.5),
            cv=3
        )
        
        # 5. 2-level Stacking (더 복잡한 구조)
        level1_models = [
            ('xgb1', xgb.XGBRegressor(max_depth=6, n_estimators=300, learning_rate=0.1, random_state=42, verbosity=0)),
            ('lgb1', lgb.LGBMRegressor(max_depth=6, n_estimators=300, learning_rate=0.1, random_state=42, verbosity=-1)),
            ('cb1', cb.CatBoostRegressor(depth=5, iterations=300, learning_rate=0.1, random_seed=42, verbose=False))
        ]
        
        level2_estimator = Ridge(alpha=0.5)
        
        self.ensemble_models['stacking_2level'] = StackingRegressor(
            estimators=level1_models,
            final_estimator=level2_estimator,
            cv=3
        )
        
        print(f"✅ Created {len(self.ensemble_models)} ensemble models")
    
    def evaluate_all_models(self, X, y, datetime_series):
        """모든 모델 평가."""
        print("📊 Evaluating all models...")
        
        ts_cv = TimeSeriesCV(n_splits=3, test_size_days=7, gap_days=1)
        temp_df = pd.DataFrame({'datetime': datetime_series})
        splits = ts_cv.split(temp_df, 'datetime')
        
        all_models = {**self.base_models, **self.ensemble_models}
        
        for model_name, model in all_models.items():
            print(f"  Evaluating {model_name}...")
            
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                try:
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    
                    # 로그 변환 역변환
                    y_val_original = np.expm1(y_val_fold)
                    y_pred_original = np.expm1(y_pred)
                    
                    fold_smape = smape(y_val_original.values, y_pred_original)
                    fold_scores.append(fold_smape)
                    
                except Exception as e:
                    print(f"    ❌ Error in fold {fold}: {e}")
                    fold_scores.append(float('inf'))
            
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            self.validation_scores[model_name] = {
                'mean': mean_score,
                'std': std_score,
                'folds': fold_scores
            }
            
            print(f"    {model_name}: {mean_score:.4f} (±{std_score:.4f})")
    
    def create_model_comparison_plot(self, output_dir='./visualizations/'):
        """모델 비교 시각화."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 데이터 준비
        model_names = list(self.validation_scores.keys())
        mean_scores = [self.validation_scores[name]['mean'] for name in model_names]
        std_scores = [self.validation_scores[name]['std'] for name in model_names]
        
        # 모델 타입별 색상
        colors = []
        for name in model_names:
            if name in ['xgboost', 'lightgbm', 'catboost']:
                colors.append('lightcoral')  # 기본 모델
            elif 'stacking' in name:
                colors.append('lightblue')   # 스태킹 모델
            else:
                colors.append('lightgreen')  # 보팅 모델
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. 평균 SMAPE 비교
        bars = ax1.bar(range(len(model_names)), mean_scores, yerr=std_scores, 
                      color=colors, alpha=0.7, capsize=5)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('SMAPE')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for i, (bar, score) in enumerate(zip(bars, mean_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2., score + std_scores[i] + 0.05,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Fold별 성능 분포
        fold_data = []
        labels = []
        for name in model_names:
            if self.validation_scores[name]['mean'] != float('inf'):
                fold_data.append(self.validation_scores[name]['folds'])
                labels.append(name)
        
        if fold_data:
            ax2.boxplot(fold_data, labels=labels)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('SMAPE')
            ax2.set_title('SMAPE Distribution Across Folds', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'advanced_ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Model comparison plot saved: {output_dir}/advanced_ensemble_comparison.png")
    
    def generate_best_ensemble_submission(self, X_train, X_test, y_train, test_df):
        """최고 성능 앙상블로 제출 파일 생성."""
        print("🎯 Generating submission with best ensemble model...")
        
        # 최고 성능 모델 선택
        best_model_name = min(self.validation_scores.keys(), 
                            key=lambda x: self.validation_scores[x]['mean'])
        best_model = {**self.base_models, **self.ensemble_models}[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (SMAPE: {self.validation_scores[best_model_name]['mean']:.4f})")
        
        # 전체 데이터로 재학습
        best_model.fit(X_train, y_train)
        
        # 테스트 예측
        test_predictions_log = best_model.predict(X_test)
        test_predictions = np.expm1(test_predictions_log)  # 로그 역변환
        test_predictions = np.maximum(test_predictions, 0)  # 음수 제거
        
        # 제출 파일 생성
        submission = pd.DataFrame({
            'num_date_time': test_df['num_date_time'],
            'answer': test_predictions
        })
        
        submission_file = f'submission_advanced_ensemble_{best_model_name}.csv'
        submission.to_csv(submission_file, index=False)
        
        print(f"💾 Submission saved: {submission_file}")
        print(f"📊 Predictions range: {test_predictions.min():.2f} - {test_predictions.max():.2f}")
        print(f"📊 Predictions mean: {test_predictions.mean():.2f}")
        
        return submission, best_model_name
    
    def create_ensemble_report(self, output_dir='./visualizations/'):
        """앙상블 결과 리포트."""
        report_path = Path(output_dir) / 'advanced_ensemble_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Advanced Ensemble Report\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | Mean SMAPE | Std SMAPE | Type |\n")
            f.write("|-------|------------|-----------|------|\n")
            
            for model_name, scores in self.validation_scores.items():
                model_type = "Base" if model_name in self.base_models else "Ensemble"
                f.write(f"| {model_name} | {scores['mean']:.4f} | {scores['std']:.4f} | {model_type} |\n")
            
            # 최고 성능 모델
            best_model = min(self.validation_scores.items(), key=lambda x: x[1]['mean'])
            f.write(f"\n## Best Performing Model\n\n")
            f.write(f"- **Model**: {best_model[0]}\n")
            f.write(f"- **SMAPE**: {best_model[1]['mean']:.4f} (±{best_model[1]['std']:.4f})\n")
            
            # 베이스라인 대비 개선
            baseline = 11.8
            improvement = (baseline - best_model[1]['mean']) / baseline * 100
            f.write(f"- **Improvement**: {improvement:.2f}% vs baseline (11.8 SMAPE)\n")
            
            f.write(f"\n## Model Types Comparison\n\n")
            
            # 기본 모델들
            f.write("### Base Models\n")
            for name in self.base_models.keys():
                if name in self.validation_scores:
                    score = self.validation_scores[name]['mean']
                    f.write(f"- **{name.title()}**: {score:.4f} SMAPE\n")
            
            # 앙상블 모델들
            f.write("\n### Ensemble Models\n")
            for name in self.ensemble_models.keys():
                if name in self.validation_scores:
                    score = self.validation_scores[name]['mean']
                    f.write(f"- **{name.replace('_', ' ').title()}**: {score:.4f} SMAPE\n")
            
            f.write(f"\n## Recommendations\n\n")
            if best_model[1]['mean'] < 10.0:
                f.write("✅ **Achieved sub-10 SMAPE target!**\n")
            elif best_model[1]['mean'] < baseline:
                f.write("✅ **Significant improvement achieved**\n")
            else:
                f.write("⚠️ **Consider further optimization**\n")
            
            f.write(f"- Use **{best_model[0]}** for final predictions\n")
            f.write(f"- Expected production SMAPE: ~{best_model[1]['mean']:.2f}\n")
        
        print(f"📄 Ensemble report saved: {report_path}")
    
    def run_advanced_ensemble(self):
        """전체 고급 앙상블 실행."""
        print("=" * 80)
        print("ADVANCED ENSEMBLE WITH CATBOOST")
        print("=" * 80)
        print("Target: Improve from current 11.8 SMAPE")
        
        # 데이터 준비
        data_dir = Path('../data')
        train_path = data_dir / 'train.csv'
        test_path = data_dir / 'test.csv'
        building_path = data_dir / 'building_info.csv'
        
        train_df, test_df = load_data(train_path, test_path, building_path)
        
        # 개선된 전처리 적용
        preprocessor = ImprovedPreprocessor()
        X_train, X_test, y_train = preprocessor.fit_transform(train_df, test_df)
        
        # datetime 시리즈 (교차검증용)
        from solution import engineer_features
        train_processed, _ = engineer_features(train_df.copy(), test_df.copy())
        datetime_series = train_processed['datetime']
        
        print(f"\n📊 Dataset: {len(X_train)} samples, {len(X_train.columns)} features")
        
        # 1. 기본 모델들 생성
        self.create_optimized_base_models()
        
        # 2. 앙상블 모델들 생성
        self.create_ensemble_models()
        
        # 3. 모든 모델 평가
        self.evaluate_all_models(X_train, y_train, datetime_series)
        
        # 4. 결과 시각화
        self.create_model_comparison_plot()
        
        # 5. 최고 성능 모델로 제출 파일 생성
        submission, best_model_name = self.generate_best_ensemble_submission(
            X_train, X_test, y_train, test_df
        )
        
        # 6. 리포트 생성
        self.create_ensemble_report()
        
        # 7. 최종 결과
        print(f"\n" + "=" * 80)
        print("ADVANCED ENSEMBLE RESULTS")
        print("=" * 80)
        
        baseline = 11.8
        best_score = min(score['mean'] for score in self.validation_scores.values())
        improvement = (baseline - best_score) / baseline * 100
        
        print(f"🎯 Best SMAPE: {best_score:.4f}")
        print(f"🚀 Improvement: {improvement:.2f}% vs baseline")
        print(f"🏆 Best model: {best_model_name}")
        
        if best_score < 9.0:
            print("🎉 Excellent! Sub-9 SMAPE achieved!")
        elif best_score < 10.0:
            print("✅ Great! Sub-10 SMAPE achieved!")
        elif best_score < baseline:
            print("✅ Good improvement over baseline!")
        
        return self.validation_scores, best_model_name


if __name__ == "__main__":
    ensemble = AdvancedEnsemble()
    results, best_model = ensemble.run_advanced_ensemble()
    
    print(f"\n🎯 Advanced ensemble completed!")
    print(f"🏆 Best model: {best_model}")
    print(f"📈 Ready for GPU server deployment!")