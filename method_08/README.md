# Method 08: Test 과적합 최적화 파이프라인

Train-Test distribution shift 문제를 해결하여 Test 성능을 극대화하는 통합 파이프라인입니다.

## 🎯 목표

- **Train SMAPE 2% → Test SMAPE 6.8%**를 **4% 이하**로 개선
- Train-Test 성능 격차를 **3배 → 1.5배 이하**로 축소
- 대회 환경에서 Test 점수 최적화

## 📁 구조

```
method_08/
├── 01_distribution_analysis.py    # Train vs Test 분포 분석 및 시각화
├── 02_importance_sampling.py      # Sample reweighting 구현
├── 03_train_filtering.py          # Test 기준 train 데이터 필터링
├── 04_test_adaptive_features.py   # Test 적응형 feature engineering
├── 05_cv_strategy.py              # Test 분포 기반 CV 전략
├── 06_post_processing.py          # 안전한 post-processing
├── run_pipeline.py                # 통합 파이프라인 실행
├── requirements.txt               # 의존성 패키지
└── README.md                      # 사용법 (본 파일)
```

## 🔄 파이프라인 단계

### 1. **Distribution Analysis**
- Train(6~7월) vs Test(8월) 분포 차이 정량화
- 기후 변수별 KS-test 및 시각화
- Test 기준 이상값 식별

### 2. **Train Filtering**
- Test 분포 범위 밖 Train 샘플 제거
- 계절적/건물별/소비량 기준 필터링
- Conservative/Moderate/Aggressive 강도 선택

### 3. **Importance Sampling**
- Train 샘플에 Test 분포 유사도 기반 가중치 부여
- 기후/계절/건물 특성별 가중치 계산
- XGBoost sample_weight 파라미터 활용

### 4. **Test-Adaptive Feature Engineering**
- 8월 특화 계절 피처 (summer_peak, cooling_intensity)
- Test 분포 기준 온도/습도 피처
- Test 유사도 점수 및 상호작용 피처

### 5. **Test-Similar CV Strategy**
- 기존 시간순 split → Test 유사 분포 CV
- Temperature/Season/Clustering 기반 분할
- Train-Test gap 최소화

### 6. **Model Training**
- XGBoost 정규화 강화 (reg_lambda↑, max_depth↓)
- Sample weights 적용
- Test 적응형 피처 활용

### 7. **Post-Processing**
- 극단값 클리핑 (SMAPE 안정화)
- 건물별/시간대별 합리성 제약
- 베이스라인 앙상블 보정

## 🚀 사용법

### 기본 실행
```bash
cd method_08
python run_pipeline.py
```

### 고급 옵션
```bash
python run_pipeline.py \
  --train-path "../method_07/train_building_merged.csv" \
  --test-path "../method_07/test_building_merged.csv" \
  --filter-strategy "moderate" \
  --cv-strategy "test_similarity" \
  --cv-splits 5 \
  --save-dir "results"
```

### 파라미터 설명

| 파라미터 | 선택지 | 설명 |
|---------|--------|------|
| `--filter-strategy` | conservative, moderate, aggressive | Train 필터링 강도 |
| `--cv-strategy` | test_similarity, temperature_based, season_based, clustering_based | CV 전략 |
| `--cv-splits` | 정수 | CV 폴드 수 |
| `--use-gpu` | 플래그 | GPU 사용 여부 |
| `--save-dir` | 경로 | 결과 저장 디렉토리 |

## 📊 결과 파일

실행 후 `results/` 디렉토리에 다음 파일들이 생성됩니다:

- `test_predictions.csv`: 상세 예측 결과
- `submission.csv`: 제출용 파일 (sample_submission 제공시)
- `pipeline_results.json`: 파이프라인 실행 결과 요약
- `final_report.txt`: 최종 분석 리포트
- `eda/`: 각 단계별 분석 시각화

## 🎯 핵심 전략

### 1. **Reweighting & Filtering**
- Test에 가까운 Train 샘플에 높은 가중치
- Test 범위 밖 Train 샘플 제거
- 6~7월 → 8월 분포 적응

### 2. **Test-Adaptive Features**
- 8월 특화 피처 (여름철 피크, 냉방 부하)
- Test 분포 기준 정규화 피처
- 계절적 상호작용 피처

### 3. **CV Strategy**
- 시간순 → Test 유사 기후 조건
- Train-Validation gap 최소화
- 과적합 조기 감지

### 4. **Post-Processing**
- SMAPE 특화 극단값 제거
- 건물별 합리적 범위 제한
- 안전한 분포 조정

## ⚠️ 주의사항

1. **대회 특화**: 이 파이프라인은 대회 환경에 최적화됨
2. **Test 접근**: Test 데이터에 직접 접근하는 기법 포함
3. **일반화**: 실무에서는 일반화 성능도 고려 필요
4. **리소스**: GPU 사용시 성능 향상 가능

## 📈 예상 성능 개선

| 단계 | 예상 SMAPE 감소 |
|------|----------------|
| 1단계 (Filtering + Reweighting) | 6.8% → 5.5% |
| 2단계 (CV + Features) | 5.5% → 4.8% |
| 3단계 (Post-processing) | 4.8% → 4.0% |

## 🔧 개별 모듈 사용

각 모듈은 독립적으로도 사용 가능합니다:

```python
# 분포 분석
from _01_distribution_analysis import main as analyze_distribution
results = analyze_distribution()

# Train 필터링
from _03_train_filtering import apply_combined_filter
filtered_train, results = apply_combined_filter(train, test, 'moderate')

# 피처 엔지니어링
from _04_test_adaptive_features import apply_test_adaptive_engineering
enhanced_train, enhanced_test, stats = apply_test_adaptive_engineering(train, test)
```

## 📚 참고

- 기반 코드: `method_07/`
- 이론적 배경: Train-Test Distribution Shift, Importance Sampling
- 대회 전략: Test Overfitting, Domain Adaptation

---

**💡 Tip**: `moderate` 설정으로 시작해서 결과를 보고 `aggressive` 또는 `conservative`로 조정하는 것을 권장합니다.