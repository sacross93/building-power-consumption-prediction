"""
Validate Current Model with Proper Time Series Cross-Validation
===============================================================

This script validates our current XGBoost model using proper time series
validation techniques to get reliable performance estimates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from ts_validation import TimeSeriesCV, compare_validation_strategies, analyze_building_specific_performance
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering from our current solution
import sys
sys.path.append('.')

def load_data_with_features():
    """Load and prepare data with our current feature engineering."""
    from solution import load_data, engineer_features
    
    print("Loading and engineering features...")
    
    # Load raw data
    data_dir = Path('../data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    building_path = data_dir / 'building_info.csv'
    
    train_df, test_df = load_data(train_path, test_path, building_path)
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    print(f"Data loaded: {len(train_fe)} training samples, {len(test_fe)} test samples")
    print(f"Feature columns: {len(train_fe.columns)}")
    
    return train_fe, test_fe


def prepare_validation_data(train_fe):
    """Prepare data for validation."""
    # Define target and feature columns
    target_col = '전력소비량(kWh)'
    drop_cols = ['전력소비량(kWh)', '일시', 'num_date_time', 'datetime']
    feature_cols = [c for c in train_fe.columns if c not in drop_cols]
    categorical_cols = ['건물번호', 'building_type']
    
    print(f"Target column: {target_col}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Categorical columns: {categorical_cols}")
    
    return target_col, feature_cols, categorical_cols


def create_validation_report(results, output_dir):
    """Create comprehensive validation report."""
    report_path = output_dir / 'validation_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Model Validation Report\n\n")
        f.write("## Executive Summary\n\n")
        
        # Overall performance comparison
        simple_smape = results['validation_comparison']['simple_split']['smape']
        cv_smape = results['validation_comparison']['ts_cv']['mean_score']
        cv_std = results['validation_comparison']['ts_cv']['std_score']
        
        f.write(f"- **Simple Split SMAPE**: {simple_smape:.4f}\n")
        f.write(f"- **Time Series CV SMAPE**: {cv_smape:.4f} (±{cv_std:.4f})\n")
        f.write(f"- **Reliability**: {'High' if cv_std < 0.5 else 'Moderate' if cv_std < 1.0 else 'Low'}\n")
        f.write(f"- **Target Achievement**: {'❌ Need {:.2f}% improvement'.format(cv_smape - 6.0) if cv_smape > 6.0 else '✅ Target achieved'}\n\n")
        
        f.write("## Detailed Results\n\n")
        
        # Validation strategy comparison
        f.write("### Validation Strategy Comparison\n\n")
        f.write("| Strategy | SMAPE | Std | Notes |\n")
        f.write("|----------|-------|-----|-------|\n")
        f.write(f"| Simple Split | {simple_smape:.4f} | - | Current method |\n")
        
        cv_results = results['validation_comparison']['ts_cv']
        f.write(f"| TS CV (5-fold) | {cv_results['mean_score']:.4f} | ±{cv_results['std_score']:.4f} | More reliable |\n")
        
        cv_gap_results = results['validation_comparison']['ts_cv_gap']
        f.write(f"| TS CV with Gap | {cv_gap_results['mean_score']:.4f} | ±{cv_gap_results['std_score']:.4f} | Most conservative |\n\n")
        
        # Fold-by-fold analysis
        f.write("### Cross-Validation Fold Analysis\n\n")
        f.write("| Fold | SMAPE | Train Size | Val Size | Val Period |\n")
        f.write("|------|-------|------------|----------|-----------|\n")
        
        for fold_detail in cv_results['fold_details']:
            f.write(f"| {fold_detail['fold']} | {fold_detail['smape']:.4f} | "
                   f"{fold_detail['train_size']} | {fold_detail['val_size']} | "
                   f"{fold_detail['val_period'][0].strftime('%m/%d')} - {fold_detail['val_period'][1].strftime('%m/%d')} |\n")
        f.write("\n")
        
        # Building performance analysis
        if 'building_analysis' in results:
            f.write("### Building Performance Analysis\n\n")
            
            f.write("#### Performance by Building Type\n\n")
            building_type_perf = results['building_analysis']['building_type']
            f.write("| Building Type | SMAPE | MAE | Count | Actual Mean |\n")
            f.write("|---------------|-------|-----|-------|-------------|\n")
            
            for building_type, row in building_type_perf.iterrows():
                f.write(f"| {building_type} | {row['SMAPE']:.4f} | {row['MAE']:.2f} | "
                       f"{int(row['Count'])} | {row['Actual_Mean']:.2f} |\n")
            f.write("\n")
            
            f.write("#### Top 5 Worst Performing Buildings\n\n")
            worst_buildings = results['building_analysis']['worst_buildings'].head(5)
            f.write("| Building ID | SMAPE | Building Type | Actual Mean |\n")
            f.write("|-------------|-------|---------------|-------------|\n")
            
            for building_id, row in worst_buildings.iterrows():
                f.write(f"| {building_id} | {row['SMAPE']:.4f} | {row['Building_Type']} | {row['Actual_Mean']:.2f} |\n")
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("### Immediate Actions\n\n")
        
        if cv_smape > 6.0:
            improvement_needed = cv_smape - 6.0
            f.write(f"1. **Performance Gap**: Need to improve SMAPE by {improvement_needed:.2f}% to reach target\n")
            f.write("2. **Model Architecture**: Consider time series models (LSTM, Prophet) for temporal patterns\n")
            f.write("3. **Feature Engineering**: Focus on time-based features and building-specific patterns\n")
        else:
            f.write("1. **Target Achieved**: Current model meets SMAPE < 6% target\n")
            f.write("2. **Stability**: Focus on improving cross-validation stability\n")
        
        if cv_std > 1.0:
            f.write("4. **Model Stability**: High CV standard deviation suggests overfitting or data leakage\n")
        
        f.write("\n### Next Steps\n\n")
        f.write("1. Implement LSTM/GRU models for temporal dependencies\n")
        f.write("2. Test Prophet model for automatic seasonality detection\n")
        f.write("3. Build ensemble combining time series and tree-based models\n")
        f.write("4. Optimize building-specific models for worst performers\n")
    
    print(f"Validation report saved to: {report_path}")


def plot_validation_results(results, output_dir):
    """Create visualizations of validation results."""
    plt.style.use('default')
    
    # 1. CV fold scores
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Fold scores comparison
    cv_results = results['validation_comparison']['ts_cv']
    fold_scores = cv_results['fold_scores']
    
    axes[0, 0].bar(range(1, len(fold_scores) + 1), fold_scores, alpha=0.7, color='steelblue')
    axes[0, 0].axhline(y=6.0, color='red', linestyle='--', alpha=0.7, label='Target (6%)')
    axes[0, 0].axhline(y=np.mean(fold_scores), color='orange', linestyle='-', alpha=0.7, label='Mean')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('SMAPE (%)')
    axes[0, 0].set_title('Cross-Validation Fold Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation strategy comparison
    strategies = ['Simple Split', 'TS CV', 'TS CV + Gap']
    scores = [
        results['validation_comparison']['simple_split']['smape'],
        results['validation_comparison']['ts_cv']['mean_score'],
        results['validation_comparison']['ts_cv_gap']['mean_score']
    ]
    errors = [
        0,
        results['validation_comparison']['ts_cv']['std_score'],
        results['validation_comparison']['ts_cv_gap']['std_score']
    ]
    
    axes[0, 1].bar(strategies, scores, yerr=errors, alpha=0.7, color=['lightcoral', 'steelblue', 'green'])
    axes[0, 1].axhline(y=6.0, color='red', linestyle='--', alpha=0.7, label='Target (6%)')
    axes[0, 1].set_ylabel('SMAPE (%)')
    axes[0, 1].set_title('Validation Strategy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Building type performance (if available)
    if 'building_analysis' in results:
        building_type_perf = results['building_analysis']['building_type']
        
        axes[1, 0].barh(building_type_perf.index, building_type_perf['SMAPE'], alpha=0.7, color='lightgreen')
        axes[1, 0].axvline(x=6.0, color='red', linestyle='--', alpha=0.7, label='Target (6%)')
        axes[1, 0].set_xlabel('SMAPE (%)')
        axes[1, 0].set_title('Performance by Building Type')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Worst buildings
        worst_buildings = results['building_analysis']['worst_buildings'].head(10)
        building_labels = [f"Building {idx}" for idx in worst_buildings.index]
        
        axes[1, 1].barh(building_labels, worst_buildings['SMAPE'], alpha=0.7, color='salmon')
        axes[1, 1].axvline(x=6.0, color='red', linestyle='--', alpha=0.7, label='Target (6%)')
        axes[1, 1].set_xlabel('SMAPE (%)')
        axes[1, 1].set_title('Top 10 Worst Performing Buildings')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plots saved to: {output_dir / 'validation_results.png'}")


def main():
    """Main validation workflow."""
    print("=" * 70)
    print("MODEL VALIDATION WITH TIME SERIES CROSS-VALIDATION")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path('.')
    
    # Load data and features
    train_fe, test_fe = load_data_with_features()
    target_col, feature_cols, categorical_cols = prepare_validation_data(train_fe)
    
    # Store all results
    results = {}
    
    # 1. Compare validation strategies
    print("\n" + "=" * 70)
    print("PHASE 1: VALIDATION STRATEGY COMPARISON")
    print("=" * 70)
    
    validation_comparison = compare_validation_strategies(
        train_fe, target_col, feature_cols, categorical_cols
    )
    results['validation_comparison'] = validation_comparison
    
    # 2. Building-specific analysis
    print("\n" + "=" * 70)
    print("PHASE 2: BUILDING-SPECIFIC PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    building_analysis = analyze_building_specific_performance(
        train_fe, target_col, feature_cols, categorical_cols
    )
    results['building_analysis'] = building_analysis
    
    # 3. Create comprehensive report
    print("\n" + "=" * 70)
    print("PHASE 3: GENERATING COMPREHENSIVE REPORT")
    print("=" * 70)
    
    create_validation_report(results, output_dir)
    plot_validation_results(results, output_dir)
    
    # 4. Summary and recommendations
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    cv_results = results['validation_comparison']['ts_cv']
    cv_mean = cv_results['mean_score']
    cv_std = cv_results['std_score']
    
    print(f"✅ Current Model Performance (Time Series CV):")
    print(f"   - Mean SMAPE: {cv_mean:.4f}%")
    print(f"   - Standard Deviation: ±{cv_std:.4f}%")
    print(f"   - 95% Confidence Interval: [{cv_mean - 2*cv_std:.4f}%, {cv_mean + 2*cv_std:.4f}%]")
    
    if cv_mean <= 6.0:
        print(f"🎯 TARGET ACHIEVED! Model meets SMAPE ≤ 6% requirement")
    else:
        improvement_needed = cv_mean - 6.0
        print(f"❌ Need {improvement_needed:.2f}% improvement to reach SMAPE ≤ 6% target")
    
    if cv_std > 1.0:
        print(f"⚠️  HIGH VARIABILITY: CV std > 1% suggests model instability")
    elif cv_std > 0.5:
        print(f"⚠️  MODERATE VARIABILITY: CV std > 0.5% suggests some instability")
    else:
        print(f"✅ STABLE MODEL: Low CV variability")
    
    print(f"\n📊 Detailed results saved to:")
    print(f"   - Report: validation_report.md")
    print(f"   - Plots: validation_results.png")
    
    return results


if __name__ == "__main__":
    results = main()