# Chee:U 직장인 스트레스 치유를 위한
# 디지털 바이오마커 활용 멘탈헬스 AI 솔루션


---

## 📋 1. 프로젝트 개요

### 1.1 기본 정보
- **프로젝트명**: Chee:U (치유)
- **서비스 슬로건**:
"스마트폰으로 간편 측정!
MBTI처럼 쉽고 재미있게 마음 상태를 알아보고 관리하는 개인 맞춤 멘탈 웰니스 앱"
(예: '평온한 나무늘보형', '조급한 토끼형’')

- **개발 기간**: 2025년 8월 4일 ~ 2025년 9월 23일
- **팀 구성**:
  - (TA, PL) 유호진
  - (AA) 김문기, 김희진
  - (DA) 김재윤, 박채윤

### 1.2 핵심 비전
- **기술적 혁신**: 스마트폰 카메라만으로 60초 만에 심박변이도(HRV) 측정
- **사용자 경험**: HRV 데이터 기반으로 정신건강 상태를 객관적으로 예측 및 분류
- **차별화 요소**: 혁신적 측정 기술 활용 / 8가지 스트레스 유형 분류 및 질문지 응답 기반 개인화된 AI 멘탈 케어

### 1.3 데이터 기반
- **학습 데이터**: 일반인 1000명 대상 수집 데이터
  - PHQ-9 (우울 9문항), GAD-7 (불안 7문항), KOSS-24 (직무스트레스 24문항)
  - 10분간 측정한 HRV 실측 데이터 (20개 특성 추출)
    - HRV 특성: HF, HR, LF, MSI, PSI, RMSSD, SDNN, SDSD, TINN, TP, calc_times, lnHF, lnLF, lnTP, meanRR, norm_HF, norm_LF, norm_VLF, pnn20, pnn50
  - 인구통계 정보 (나이, 성별, NCS 표준직군)
- **측정 기술**: rPPG (Remote Photoplethysmography) SDK 활용

---

## 🔄 2. 서비스 플로우

### 2.1 전체 사용자 여정
```
1. 회원가입 → 2. HRV 측정 + 설문 응답 → 3. AI 분석 → 4. 통합 프로필 생성 → 5. 맞춤형 치유 캡슐 제공
```

### 2.2 단계별 상세 플로우

#### **1단계: 회원가입 및 프로필 설정**
- Google OAuth 2.0 인증
- 개인정보 수집: 성별, 연령, 직군(NCS 분류)
- Firebase Authentication 연동

#### **2단계: HRV 측정 및 설문 응답 (첫 측정 시)**
- **HRV 측정**: 60초간 전면 카메라로 얼굴 촬영
  - 추출 데이터: 20개 HRV 특성 (시간/주파수/비선형 도메인)
- **설문 응답** (첫 측정 시 동시 진행):
  - PHQ-9 (우울 9문항)
  - GAD-7 (불안 7문항)
  - KOSS-24 (직무스트레스 24문항)
  - 총 40문항 응답

#### **3단계: AI 분석 및 분류**
- **분석 프로세스**: HRV 데이터 → CheeU AI 모델 → 설문 점수 예측 → 바이너리 분류
- **예측 모델**: Deep Neural Network / LightGBM
- **예측 대상**:
  - PHQ-9 점수 (0-27): 우울 수준 예측
  - GAD-7 점수 (0-21): 불안 수준 예측
  - KOSS-24 점수 (0-100): 직무 스트레스 예측
- **바이너리 분류**: 예측 점수를 임계값 기준으로 각 카테고리별 Yes/No 결정
- **참고**: 실제 설문 응답은 분류에 직접 사용되지 않음

#### **4단계: 통합 프로필 생성**
- **스트레스 유형 결정**: 3개 바이너리 값 조합 → 8가지 유형 (2³) + 고유 캐릭터
- **개인 특징 추출**: 설문 키워드 분석 → 개인별 핵심 특성 도출
- **통합 프로필**: 스트레스 유형 × 개인 특징 → 고유한 개인화 프로필 생성
- **치료 방향 설정**: 통합 프로필 기반 맞춤형 치유 전략 수립

### 2.3 설문지 활용 전략
설문지가 스트레스 유형 분류에 직접 사용되지 않는다면 왜 수집하는가?

#### **1. 사용자 신뢰도 향상**
- 유형 분류 과정에 대한 사용자의 신뢰성 증대
- 투명한 측정 과정으로 서비스 수용성 향상

#### **2. 응답 정직성 증진**
- HRV 측정과 설문 응답을 동시에 진행
- 생체신호 측정 중 설문에 응답함으로써 정직한 답변 유도
- 의식적 조작이 어려운 환경에서의 솔직한 자기보고

#### **3. 개인화된 치유 콘텐츠 생성**
- **키워드 추출**: 설문 응답 중 2-3점 체크 항목에서 핵심 키워드 식별
- **개인 특징 도출**: 설문 유형(PHQ-9/GAD-7/KOSS-24) + 고득점 키워드 조합
- **맞춤형 치유 캡슐**: 추출된 개인 특징을 반영한 심리학 기반 치료 요법 제공

#### **5단계: 맞춤형 치유 캡슐 제공 (마음 CheeU 봇)**
- **통합 개인화 시스템**: 
  - **HRV 분석 기반**: 스트레스 유형 (8가지 캐릭터 유형)
  - **설문 키워드 기반**: 개인 특징 (2-3점 체크 항목에서 추출된 키워드)
  - **통합 프로파일**: 스트레스 유형 × 개인 특징 → 고유한 맞춤형 프로필 생성

- **치유 캡슐 생성 프로세스**:
  - **1차 필터링**: 스트레스 유형에 따른 기본 치료 요법 선택
  - **2차 개인화**: 설문 키워드를 활용한 세부 맞춤화
  - **동적 생성**: RAG + LangChain + GPT-4o로 실시간 개인 맞춤 콘텐츠 생성

- **기술 스택**: 
  - VectorDB (심리학 기반 치료 요법 데이터베이스)
  - RAG (개인 특징 기반 검색 증강 생성)
  - LangChain (통합 프로파일 처리 파이프라인)
  - GPT-4o (개인화된 치유 캡슐 생성)

---

## 🏗️ 3. 시스템 아키텍처

### 3.1 기술 스택

#### **Frontend (Android Native)**
- **개발 환경**: Android Studio (Kotlin)
- **UI Framework**: Jetpack Compose
- **아키텍처 패턴**: MVVM + Clean Architecture
- **주요 라이브러리**:
  - Hilt (의존성 주입)
  - Navigation Compose (화면 전환)
  - CameraX (카메라 제어)
  - Coroutines + Flow (비동기 처리)

#### **Backend (Firebase Ecosystem)**
- **인증**: Firebase Authentication (Google OAuth)
- **데이터베이스**: Cloud Firestore (NoSQL)
- **서버리스**: Cloud Functions (Python/Node.js)
- **저장소**: Cloud Storage (미디어 파일)
- **ML 서빙**: Firebase ML / Cloud Functions

#### **AI/ML Pipeline**
- **HRV 분석**: DeepMedi SDK (rPPG 측정)
- **분류 모델**: LightGBM (Python)
- **벡터 DB**: Firestore Vector Extensions
- **LLM 통합**: LangChain + OpenAI GPT-4o

### 3.2 데이터 플로우
```
Android App → Camera Capture + Questionnaire → rPPG Processing → HRV Extraction
    ↓                                             ↓
    └── Questionnaire Responses ──────────────────┘
    ↓
Firebase Functions → ML Model (HRV → Score Prediction) → Binary Classification
    ↓                    ↓
    ↓                    └── Extract Keywords from High-Score Items (2-3점)
    ↓
Firestore → Store Results → Stress Type + Personal Keywords → Integrated Profile
    ↓
RAG Pipeline → 1차 필터링 (스트레스 유형) → 2차 개인화 (키워드 기반)
    ↓
LLM Generation → 맞춤형 치유 캡슐 → Personalized Healing Content

```

### 3.3 주요 컴포넌트

#### **Android App 구조**
```
com.cheeu.app/
├── demo/
│   ├── ui/
│   │   ├── auth/         # 인증 및 프로필
│   │   ├── home/         # 홈 화면
│   │   ├── measure/      # HRV 측정
│   │   └── onboarding/   # 온보딩
│   ├── data/
│   │   └── api/          # API 통신
│   └── di/               # 의존성 주입
├── module/
│   ├── camera/           # 카메라 모듈
│   ├── deepmedi/         # HRV SDK 연동
│   └── hrv/              # HRV 매니저
└── callbacks/            # 콜백 인터페이스
```

#### **Backend Services**
- **Cloud Functions 엔드포인트**:
  - `/analyzeHRV`: HRV 데이터 분석 및 점수 예측
  - `/extractKeywords`: 설문 응답에서 키워드 추출 (2-3점 항목)
  - `/createIntegratedProfile`: 스트레스 유형 + 개인 특징 통합 프로필 생성
  - `/generateHealingCapsule`: 맞춤형 치유 캡슐 생성 (RAG + LLM)
  - `/chatWithBot`: 개인화된 챗봇 대화 처리

---

## 🎭 4. 8가지 스트레스 유형 시스템

### 4.1 분류 매트릭스

| 코드 | 우울 | 불안 | 직무스트레스 | 유형명 | 캐릭터 |
|------|------|------|-------------|--------|---------|
| XXX | No | No | No | 평온형 | 🦥 평온한 나무늘보형 |
| OXX | Yes | No | No | 우울형 | 🐻 겨울잠 자는 곰형 |
| XOX | No | Yes | No | 불안형 | 🐰 조급한 토끼형 |
| XXO | No | No | Yes | 직무스트레스형 | 🦔 가시돋친 고슴도치형 |
| OOX | Yes | Yes | No | 우울+불안형 | 🦌 고뇌하는 사슴형 |
| OXO | Yes | No | Yes | 우울+직무스트레스형 | 🦫 지친 비버형 |
| XOO | No | Yes | Yes | 불안+직무스트레스형 | 🐿️ 바쁜 다람쥐형 |
| OOO | Yes | Yes | Yes | 위기형 | 🦊 혼란스런 여우형 |

### 4.2 유형별 특성 및 대응 전략
> **논문 기반 신뢰성 검증 완료** - VectorDB 내 10개 심리학 논문 분석 결과

#### **1️⃣ 평온형 (🦥 평온한 나무늘보형) - XXX**
**📊 논문 기반 특성**
- 정신건강 상태 양호, 자연적 회복력 보유
- HRV 특성: 균형잡힌 자율신경계 활동
- 기준선 유지 능력 우수

**🎯 증거 기반 대응 전략: MBSR (마음챙김 기반 스트레스 감소)**
- **치료 목표**: 현재 상태 유지 및 예방적 관리
- **핵심 기법**: 마음챙김 명상(일일 10-20분), 자기돌봄 루틴, 스트레스 조기 신호 인식
- **효과 근거**: "스트레스로 인한 불편한 증상 완화, 행복·긍정적 정동 향상" (Hayes, 1994)

#### **2️⃣ 우울형 (🐻 겨울잠 자는 곰형) - OXX**
**📊 논문 기반 특성**
- 핵심 증상: 우울감 지속, 무기력, 의욕 저하
- 인지 패턴: 부정적 자동사고, 자기비판적 사고
- 행동 패턴: 활동 감소, 사회적 철수

**🎯 증거 기반 대응 전략: 긍정심리치료 (Positive Psychotherapy)**
- **치료 목표**: 행복 증진 (증상 감소가 아닌 긍정 중심)
- **핵심 기법**: 성격강점 활용하기, 축복하기(Three Good Things), 미래 최고의 모습 떠올리기, 향유(Savoring)
- **효과 근거**: "삶의 만족도·긍정정서 유의한 증가, 우울증상·부정정서 유의한 감소" (임영진, 2023)

#### **3️⃣ 불안형 (🐰 조급한 토끼형) - XOX**
**📊 논문 기반 특성**
- 핵심 증상: 불안 수준 높음, 긴장과 걱정 지속
- 인지 왜곡: "사람들이 나만 주목한다", "실수하면 부정적 평가받을 것", "무능하게 보이는 것은 최악"
- 회피 행동: 사회적 상황 회피, 완벽주의적 대처

**🎯 증거 기반 대응 전략: 인지행동치료 (CBT) - 사회불안장애 치료법**
- **치료 목표**: 왜곡된 인지를 합리적으로 바꾸어 정서·행동 변화 유도
- **핵심 기법**: 인지적 재구성, 노출 훈련, 행동실험, 점진적 이완훈련
- **효과 근거**: "SAD에 대한 CBT의 효능과 효과는 충분한 연구를 통해 밝혀져 있다" (이재현, 2016)

#### **4️⃣ 직무스트레스형 (🦔 가시돋친 고슴도치형) - XXO**
**📊 논문 기반 특성**
- 핵심 증상: 높은 업무 스트레스, 정서적 안정성은 유지
- 스트레스 원인: 업무 과부하, 시간 압박, 역할 갈등
- 신체 반응: HRV 불균형, 자율신경계 과활성

**🎯 증거 기반 대응 전략: MBSR + 직무 스트레스 관리**
- **치료 목표**: 스트레스 완화 및 자기효능감 향상
- **핵심 기법**: 마음챙김 기반 스트레스 감소, 시간관리·우선순위 설정, 경계설정(Work-Life Balance), 자기돌봄 전략
- **효과 근거**: "직무 스트레스 및 심리적 소진 해소에 긍정적 영향" (MBSR 연구, 2023)

#### **5️⃣ 우울+불안형 (🦌 고뇌하는 사슴형) - OOX**
**📊 논문 기반 특성**
- 복합 증상: 우울과 불안 공존, 복합적 정서 문제
- 심리적 경직성: 부정적 경험 회피, 융합된 사고
- 기능 손상: 일상생활 전반의 어려움

**🎯 증거 기반 대응 전략: 수용전념치료 (ACT) - 심리적 유연성 접근**
- **치료 목표**: 심리적 경직성 → 심리적 유연성
- **ACT 6가지 핵심 과정**: 수용(Acceptance), 탈융합(Defusion), 현재 순간(Present Moment), 맥락으로서의 자기(Self-as-Context), 가치(Values), 전념행동(Committed Action)
- **효과 근거**: "불안장애와 우울장애 영역에서 무작위 대조군 연구를 통한 임상 효과 보고" (ACT 연구, 2023)

#### **6️⃣ 우울+직무스트레스형 (🦫 지친 비버형) - OXO**
**📊 논문 기반 특성**
- 핵심 증상: 정서적 탈진, 직업적 효능감 저하
- 번아웃 증후군: 신체적·정서적 고갈 상태
- 에너지 고갈: 지속적 피로감, 동기 상실

**🎯 증거 기반 대응 전략: 통합적 접근 (긍정심리치료 + MBSR)**
- **치료 목표**: 번아웃 회복 및 에너지 관리
- **핵심 기법**: 에너지 관리·회복 시간 확보, 의미 찾기·일의 가치 재발견, 자기연민(Self-Compassion), 점진적 활동 증가
- **효과 근거**: 긍정심리치료 "우울증상 현저한 호전, 삶의 만족도·긍정정서 증가" + MBSR 자기돌봄 효과

#### **7️⃣ 불안+직무스트레스형 (🐿️ 바쁜 다람쥐형) - XOO**
**📊 논문 기반 특성**
- 핵심 증상: 높은 스트레스와 불안, 번아웃 위험
- 성과 불안: 완벽주의, 실패에 대한 두려움
- 과활성화: 지속적 긴장 상태, 휴식 불능

**🎯 증거 기반 대응 전략: CBT + ACT 통합 접근**
- **치료 목표**: 통합적 스트레스 관리, 우선순위 설정
- **핵심 기법**: 인지재구성(완벽주의 사고 교정), 가치 기반 우선순위(ACT), 점진적 노출(불완전함), 경계 설정(과도한 책임감 조절)
- **효과 근거**: CBT의 인지재구성 + ACT의 가치 지향적 행동 활성화 효과

#### **8️⃣ 위기형 (🦊 혼란스런 여우형) - OOO**
**📊 논문 기반 특성**
- 고위험 상태: 모든 영역(우울, 불안, 직무스트레스) 임계점
- 즉각적 개입 필요: 심각한 기능 손상
- 복합적 어려움: 다중 스트레스원, 대처 자원 부족

**🎯 증거 기반 대응 전략: 위기 개입 + 통합적 치료 (DBT 요소 포함)**
- **치료 목표**: 안정화 → 단계적 회복
- **위기 대응**: 즉각적 안정화·안전 계획 수립, 전문가 상담 연결·정신건강 전문의 협진, 사회적 지지망 활성화
- **중장기 치료**: 변증법적 행동치료(DBT) 기술 활용, 통합적 접근(CBT+ACT+긍정심리치료 병행)
- **효과 근거**: "변증법적 행동치료 기술훈련이 대학생 사회불안 감소에 효과" + 위기 개입 프로토콜

---

#### **📚 신뢰도 검증 요소**
**논문 근거 강도**
- RCT (무작위 대조군 연구) 기반 효과성 입증
- 메타분석 결과 포함 (MBSR, ACT, CBT)
- 한국 문화적 맥락 고려 (K-MBSR, 한국형 CBT)

**임상적 타당성**
- 표준화된 평가도구 사용 (PHQ-9, GAD-7, KOSS-24)
- 치료 효과 지속성 확인 (1개월-1년 추적 관찰)
- 다양한 연령층 검증 (대학생, 성인, 노인)

---


## 📊 6. 데이터 모델

### 6.1 주요 컬렉션 구조

#### **Users Collection**
```json
{
  "userId": "string",
  "email": "string",
  "profile": {
    "name": "string",
    "age": "number",
    "gender": "string",
    "occupation": "string"
  },
  "currentStressType": "string",
  "currentCharacter": "string"
}
```

#### **Measurements Collection**
```json
{
  "measurementId": "string",
  "userId": "string",
  "timestamp": "timestamp",
  "isInitialMeasurement": "boolean",
  "hrvMetrics": {
    "HR": "number",
    "RMSSD": "number",
    "SDNN": "number",
    "LF": "number",
    "HF": "number",
    "LF_HF_Ratio": "number"
  },
  "questionnaireResponses": {
    "phq9Answers": "array[number]",
    "gad7Answers": "array[number]",
    "koss24Answers": "array[number]",
    "extractedKeywords": "array[string]",
    "highScoreItems": {
      "phq9": "array[string]",
      "gad7": "array[string]",
      "koss24": "array[string]"
    }
  },
  "predictions": {
    "phq9Score": "number",
    "gad7Score": "number",
    "koss24Score": "number"
  },
  "classification": {
    "depression": "boolean",
    "anxiety": "boolean",
    "occupationalStress": "boolean",
    "stressType": "string"
  }
}
```

#### **PersonalizationProfiles Collection**
```json
{
  "profileId": "string",
  "userId": "string",
  "stressType": "string",
  "personalKeywords": "array[string]",
  "dominantQuestionnaireType": "string", // PHQ-9, GAD-7, KOSS-24
  "highScoreKeywords": "array[string]",
  "therapyPreferences": "object",
  "lastUpdated": "timestamp"
}
```

---


## 🔒 7. 보안 및 규제

### 9.1 데이터 보호
- **암호화**: TLS 1.3 전송, AES-256 저장
- **인증**: JWT 토큰 기반
- **익명화**: 개인정보 마스킹

### 9.2 규제 준수
- **개인정보보호법**: GDPR, KISA 가이드라인
- **의료기기 규제**: 웰니스 기기 포지셔닝
- **면책 조항**: 의료 진단 목적 아님 명시

---

## 💡 8. 핵심 인사이트

1. **이중 전략**: HRV 기반 객관적 분류 + 설문 기반 개인화 콘텐츠
2. **신뢰성 향상**: 동시 측정을 통한 사용자 신뢰도 및 응답 정직성 증대
3. **동적 개인화**: 실시간 키워드 추출로 고정 콘텐츠를 넘어선 맞춤형 치료
4. **기술 융합**: rPPG + 심리학 + AI + RAG의 혁신적 결합
5. **지속가능성**: 사용자 특성 학습을 통한 지속적 개선

---

*작성일: 2025년 9월 11일*
*프로젝트: K-Digital Training 데이터 기반 차세대 디지털 헬스케어 AI 솔루션 과정*
*문서 버전: 1.0*
