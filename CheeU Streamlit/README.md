# CheeU MVP - Mental Health Analysis Service

CheeU MVP는 심박변이도(HRV) 데이터와 설문 점수를 활용하여 우울 및 불안 상태를 예측하는 Streamlit 웹 애플리케이션입니다.

## 📋 주요 기능

- **바이너리 분류**: PHQ-9 기반 우울증 예측 (Yes/No)
- **바이너리 분류**: GAD-7 기반 불안장애 예측 (Yes/No)
- **실제 데이터**: 테스트 데이터에서 추출한 12개의 실제 샘플
- **HRV 분석**: 20개 심박변이도 특성값 활용
- **메타데이터 표시**: 25개 변수 (HRV 20개 + 설문 2개 + 인구통계 3개)

## 🚀 배포 방법

### Streamlit Community Cloud

1. GitHub 리포지토리에 업로드
2. [Streamlit Community Cloud](https://share.streamlit.io/)에서 배포
3. 메인 파일: `app.py`

### 로컬 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# 애플리케이션 실행
streamlit run app.py
```

## 📁 파일 구조

```
/
├── app.py                          # 메인 Streamlit 애플리케이션
├── requirements.txt                # Python 패키지 의존성
├── README.md                      # 프로젝트 설명서
├── test_extracted_samples.csv     # 테스트 데이터 샘플 (12개)
├── LGBM_GAD_model.pkl            # GAD-7 예측 모델
├── LGBM_PHQ_model.pkl            # PHQ-9 예측 모델
└── .streamlit/
    └── config.toml               # Streamlit 설정
```

## 🔬 모델 정보

- **GAD 모델**: LightGBM 기반 불안장애 예측 (F1-score: 0.50-0.56)
- **PHQ 모델**: LightGBM 기반 우울증 예측 (F1-score: 0.62)
- **입력 데이터**: 25개 특성값 (HRV 20개 + 설문점수 2개 + 인구통계 3개)

## 📊 샘플 데이터

총 12개 샘플, 4개 케이스별 3개씩:
- **😌 정상형**: 우울/불안 증상이 없는 실제 데이터
- **😰 불안형**: 불안 증상만 있는 실제 데이터  
- **😢 우울형**: 우울 증상만 있는 실제 데이터
- **🌀 우울+불안 복합형**: 우울과 불안 증상이 모두 있는 실제 데이터

## 🎯 판정 기준

- **PHQ-9 ≥ 10**: 우울증 양성 (Yes)
- **GAD-7 ≥ 10**: 불안장애 양성 (Yes)
- **10점 미만**: 음성 (No)

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **ML**: LightGBM, scikit-learn
- **Data**: pandas, numpy
- **언어**: Python 3.8+

## 📝 사용법

1. 웹 애플리케이션 접속
2. 드롭다운에서 분석할 샘플 선택
3. 샘플의 메타데이터 확인
4. AI 모델 예측 결과 확인 (우울/불안 Yes/No)
5. 신뢰도 점수 확인

---

*이 프로젝트는 CheeU 팀의 MVP(Minimum Viable Product)로 개발되었습니다.*