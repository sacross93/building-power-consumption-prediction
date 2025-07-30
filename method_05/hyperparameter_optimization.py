"""
Advanced Hyperparameter Optimization
====================================

Optuna를 사용한 체계적 하이퍼파라미터 최적화
현재 11.8 SMAPE에서 추가 5-10% 개선 목표
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import VotingRegressor
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from solution import load_data, smape
from ts_validation import TimeSeriesCV
from improved_preprocessing import ImprovedPreprocessor


class HyperparameterOptimizer:
    """하이퍼파라미터 최적화 클래스."""
    
    def __init__(self, n_trials=100, cv_folds=3):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = {}
        self.study_results = {}
        
    def optimize_xgboost(self, X, y, datetime_series):
        """XGBoost 하이퍼파라미터 최적화."""
        print("🔍 Optimizing XGBoost hyperparameters...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'random_state': 42,
                'verbosity': 0
            }
            
            return self._cross_validate_model(xgb.XGBRegressor(**params), X, y, datetime_series)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='xgboost_optimization'
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params['xgboost'] = study.best_params
        self.study_results['xgboost'] = study
        
        print(f"✅ XGBoost best SMAPE: {study.best_value:.4f}")
        print(f"📊 Best params: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def optimize_lightgbm(self, X, y, datetime_series):
        """LightGBM 하이퍼파라미터 최적화."""
        print("🔍 Optimizing LightGBM hyperparameters...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'random_state': 42,
                'verbosity': -1
            }
            
            return self._cross_validate_model(lgb.LGBMRegressor(**params), X, y, datetime_series)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='lightgbm_optimization'
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params['lightgbm'] = study.best_params
        self.study_results['lightgbm'] = study
        
        print(f"✅ LightGBM best SMAPE: {study.best_value:.4f}")
        print(f"📊 Best params: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def optimize_catboost(self, X, y, datetime_series):
        """CatBoost 하이퍼파라미터 최적화."""
        print("🔍 Optimizing CatBoost hyperparameters...")
        
        def objective(trial):
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'iterations': trial.suggest_int('iterations', 200, 1000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                'max_leaves': trial.suggest_int('max_leaves', 10, 1000),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'random_seed': 42,
                'verbose': False
            }
            
            # Bootstrap type에 따른 추가 파라미터
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample_bernoulli', 0.1, 1.0)
            
            return self._cross_validate_model(cb.CatBoostRegressor(**params), X, y, datetime_series)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='catboost_optimization'
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params['catboost'] = study.best_params
        self.study_results['catboost'] = study
        
        print(f"✅ CatBoost best SMAPE: {study.best_value:.4f}")
        print(f"📊 Best params: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def optimize_ensemble_weights(self, X, y, datetime_series):
        """앙상블 가중치 최적화."""
        print("🔍 Optimizing ensemble weights...")
        
        # 최적화된 개별 모델들 생성
        xgb_model = xgb.XGBRegressor(**self.best_params['xgboost'])
        lgb_model = lgb.LGBMRegressor(**self.best_params['lightgbm'])
        cb_model = cb.CatBoostRegressor(**self.best_params['catboost'])
        
        def objective(trial):
            # 가중치 제약: 합이 1이 되도록
            w1 = trial.suggest_float('weight_xgb', 0.1, 0.8)
            w2 = trial.suggest_float('weight_lgb', 0.1, 0.8)
            w3 = 1.0 - w1 - w2
            
            if w3 < 0.1 or w3 > 0.8:
                return float('inf')  # 제약 조건 위반
            
            ensemble = VotingRegressor([
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('cb', cb_model)
            ], weights=[w1, w2, w3])
            
            return self._cross_validate_model(ensemble, X, y, datetime_series)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='ensemble_weights'
        )
        
        study.optimize(objective, n_trials=50, show_progress_bar=True)  # 더 적은 trials
        
        self.best_params['ensemble_weights'] = study.best_params
        self.study_results['ensemble_weights'] = study
        
        print(f"✅ Ensemble best SMAPE: {study.best_value:.4f}")
        print(f"📊 Best weights: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def _cross_validate_model(self, model, X, y, datetime_series):
        """모델 교차검증."""
        ts_cv = TimeSeriesCV(n_splits=self.cv_folds, test_size_days=7, gap_days=1)
        temp_df = pd.DataFrame({'datetime': datetime_series})
        splits = ts_cv.split(temp_df, 'datetime')
        
        scores = []
        for train_idx, val_idx in splits:
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            try:
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                # 로그 변환된 타겟 역변환
                y_val_original = np.expm1(y_val_fold)
                y_pred_original = np.expm1(y_pred)
                
                fold_smape = smape(y_val_original.values, y_pred_original)
                scores.append(fold_smape)
                
            except Exception as e:
                return float('inf')  # 학습 실패시 최대값 반환
        
        return np.mean(scores)
    
    def create_optimization_plots(self, output_dir='./visualizations/'):
        """최적화 결과 시각화."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (model_name, study) in enumerate(self.study_results.items()):
            if i >= 4:  # 최대 4개 모델만 표시
                break
                
            # 최적화 과정 시각화
            trials_df = study.trials_dataframe()
            
            axes[i].plot(trials_df['number'], trials_df['value'], alpha=0.6)
            axes[i].axhline(y=study.best_value, color='red', linestyle='--', 
                          label=f'Best: {study.best_value:.4f}')
            axes[i].set_title(f'{model_name.title()} Optimization', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Trial')
            axes[i].set_ylabel('SMAPE')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hyperparameter_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Optimization plots saved: {output_dir}/hyperparameter_optimization.png")
    
    def create_optimization_report(self, output_dir='./visualizations/'):
        """최적화 결과 리포트."""
        report_path = Path(output_dir) / 'hyperparameter_optimization_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Hyperparameter Optimization Report\n\n")
            
            f.write("## Optimization Results\n\n")
            
            for model_name, study in self.study_results.items():
                f.write(f"### {model_name.title()}\n")
                f.write(f"- **Best SMAPE**: {study.best_value:.4f}\n")
                f.write(f"- **Trials completed**: {len(study.trials)}\n")
                f.write(f"- **Best parameters**:\n")
                for param, value in study.best_params.items():
                    f.write(f"  - `{param}`: {value}\n")
                f.write("\n")
            
            # 성능 비교
            f.write("## Performance Comparison\n\n")
            f.write("| Model | Best SMAPE | Improvement |\n")
            f.write("|-------|------------|-------------|\n")
            
            baseline_smape = 11.8  # 현재 성능
            for model_name, study in self.study_results.items():
                improvement = (baseline_smape - study.best_value) / baseline_smape * 100
                f.write(f"| {model_name.title()} | {study.best_value:.4f} | {improvement:+.2f}% |\n")
            
            f.write("\n## Recommendations\n\n")
            best_model = min(self.study_results.items(), key=lambda x: x[1].best_value)
            f.write(f"- **Best performing model**: {best_model[0].title()}\n")
            f.write(f"- **Best SMAPE**: {best_model[1].best_value:.4f}\n")
            f.write(f"- **Recommended for production**: Use optimized {best_model[0]} parameters\n")
        
        print(f"📄 Optimization report saved: {report_path}")
    
    def run_full_optimization(self):
        """전체 최적화 실행."""
        print("=" * 80)
        print("ADVANCED HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"Target: Improve from current 11.8 SMAPE")
        print(f"Trials per model: {self.n_trials}")
        
        # 데이터 준비
        data_dir = Path('../data')
        train_path = data_dir / 'train.csv'
        test_path = data_dir / 'test.csv'
        building_path = data_dir / 'building_info.csv'
        
        train_df, test_df = load_data(train_path, test_path, building_path)
        
        # 개선된 전처리 적용
        preprocessor = ImprovedPreprocessor()
        X, X_test, y = preprocessor.fit_transform(train_df, test_df)
        
        # datetime 시리즈 생성 (교차검증용)
        from solution import engineer_features
        train_processed, _ = engineer_features(train_df.copy(), test_df.copy())
        datetime_series = train_processed['datetime']
        
        # 1. 개별 모델 최적화
        print(f"\n🚀 Starting optimization with {len(X)} samples, {len(X.columns)} features")
        
        xgb_params, xgb_score = self.optimize_xgboost(X, y, datetime_series)
        lgb_params, lgb_score = self.optimize_lightgbm(X, y, datetime_series)
        cb_params, cb_score = self.optimize_catboost(X, y, datetime_series)
        
        # 2. 앙상블 가중치 최적화
        ensemble_params, ensemble_score = self.optimize_ensemble_weights(X, y, datetime_series)
        
        # 3. 결과 시각화 및 리포트
        self.create_optimization_plots()
        self.create_optimization_report()
        
        # 4. 최종 결과
        print(f"\n" + "=" * 80)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)
        
        baseline = 11.8
        results = [
            ('XGBoost', xgb_score),
            ('LightGBM', lgb_score),
            ('CatBoost', cb_score),
            ('Ensemble', ensemble_score)
        ]
        
        for name, score in results:
            improvement = (baseline - score) / baseline * 100
            print(f"{name:12}: {score:.4f} SMAPE ({improvement:+.2f}% vs baseline)")
        
        best_score = min(score for _, score in results)
        best_improvement = (baseline - best_score) / baseline * 100
        
        print(f"\n🎯 Best result: {best_score:.4f} SMAPE")
        print(f"🚀 Total improvement: {best_improvement:.2f}%")
        
        if best_score < 10.0:
            print("✅ Achieved sub-10 SMAPE target!")
        
        return self.best_params, results


if __name__ == "__main__":
    # 빠른 테스트용으로 trials 수를 줄임
    optimizer = HyperparameterOptimizer(n_trials=30, cv_folds=2)  # 실제로는 100, 3으로 설정
    
    best_params, results = optimizer.run_full_optimization()
    
    print(f"\n🎯 Hyperparameter optimization completed!")
    print(f"📈 Ready for GPU server deployment with optimized parameters!")