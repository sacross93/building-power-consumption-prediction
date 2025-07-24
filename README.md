# Building Power Consumption Prediction

건물별 전력 소비량 예측을 위한 머신러닝 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 건물의 특성과 날씨 데이터를 활용하여 전력 소비량을 예측하는 시스템입니다. LightGBM을 사용하여 다양한 건물 유형(호텔, 학교, 병원, 상용, 아파트 등)별로 전력 소비 패턴을 학습하고 예측합니다.

## 주요 기능

- **피처 엔지니어링**: 시간, 날씨, 공휴일, 건물 특성 등을 활용한 고급 피처 생성
- **건물 유형별 모델링**: 각 건물 유형에 맞는 개별 모델 학습
- **시계열 예측**: 과거 데이터를 활용한 전력 소비량 예측
- **SMAPE 평가**: 대칭 평균 절대 백분율 오차를 통한 모델 성능 평가

## 데이터 구조

- `train.csv`: 학습용 데이터 (과거 전력 소비량 포함)
- `test.csv`: 테스트용 데이터 (예측 대상)
- `building_info.csv`: 건물 정보 (유형, 면적, 태양광 용량 등)
- `sample_submission.csv`: 제출 형식 예시

## 사용된 기술

- **Python 3.x**
- **pandas**: 데이터 처리
- **LightGBM**: 그래디언트 부스팅 모델
- **numpy**: 수치 계산
- **holidays**: 공휴일 정보

## 설치 및 실행

1. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:
```bash
pip install pandas lightgbm numpy holidays
```

3. 모델 실행:
```bash
python test.py
```

## 피처 설명

### 시간 관련 피처
- 월, 일, 시간, 요일
- 주말 여부
- 시간의 사인/코사인 변환

### 날씨 관련 피처
- 기온, 풍속, 습도
- 체감온도(THI)
- 기온과 시간의 상호작용
- 기온의 이동평균 및 표준편차

### 건물 특성 피처
- 건물 유형
- 연면적, 냉방면적
- 태양광 용량, ESS 저장용량, PCS 용량

### 시차 변수
- 24시간, 48시간, 168시간(1주) 전의 전력 소비량 및 기온

## 모델 성능

- **평가 지표**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **모델**: LightGBM (Light Gradient Boosting Machine)
- **최적화**: Early Stopping을 통한 과적합 방지

## 파일 구조

```
├── test.py              # 메인 실행 파일
├── data/                # 데이터 디렉토리
│   ├── train.csv       # 학습 데이터
│   ├── test.csv        # 테스트 데이터
│   ├── building_info.csv # 건물 정보
│   └── sample_submission.csv # 제출 형식
├── submission.csv       # 예측 결과
├── .gitignore          # Git 무시 파일
└── README.md           # 프로젝트 설명서
```

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 