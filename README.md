# CheeU Chatbot RAG Pipeline

## 🎯 주요 개선사항 (v2.0)

### ✅ **한국어 지원 대폭 향상**
- **임베딩 모델 변경**: `paraphrase-multilingual-MiniLM-L12-v2` (다국어 지원)
- **검색 정확도 향상**: 한국어 텍스트 의미론적 유사도 개선
- **VectorDB 최적화**: 419개 한국어 연구논문 완전 지원

### ✅ **실제 검증 완료**
- **이대리 페르소나 테스트** 통과
- **VectorDB 연동** 정상 작동
- **프롬프트 생성** 완전 검증
- **gpt-5-nano 모델** 지원

## 📦 프로젝트 구조

```
CheeU-Final-Release/
├── cheeu_rag/                # 핵심 모듈
│   ├── __init__.py           # API 엔트리포인트
│   ├── models.py             # 데이터 모델 정의
│   ├── pipeline.py           # 메인 파이프라인
│   ├── chatbot.py            # LLM 챗봇 엔진
│   ├── vectordb.py           # VectorDB 관리 (다국어 모델)
│   └── api.py                # 간편 API 함수
├── data/vectordb/            # VectorDB 데이터 (419개 논문)
├── streamlit_app.py          # 웹 UI 앱
├── test_simple.py            # 간단 테스트
├── requirements.txt          # 의존성
├── .env.example              # 환경변수 예시
├── ARCHITECTURE.md           # 시스템 아키텍처
└── README.md                 # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정
```bash
# .env 파일 생성
cp .env.example .env

# .env 파일에 OpenAI API 키 입력
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. 실행
```bash
# Streamlit 앱 실행
streamlit run streamlit_app.py

# 또는 간단 테스트
python test_simple.py
```

## 🏗️ 시스템 아키텍처

### 파이프라인 흐름
```
사용자 입력 → 1차 필터링 (스트레스 유형) → 2차 필터링 (개인화) → LLM 생성 → 치유메시지
```

### 핵심 개선사항

#### 1. **향상된 임베딩 모델**
- **이전**: `all-MiniLM-L6-v2` (영어 최적화)
- **현재**: `paraphrase-multilingual-MiniLM-L12-v2` (다국어 지원)
- **결과**: 한국어 검색 정확도 대폭 향상

#### 2. **스트레스 유형 분류** (8가지)
- XXX = 평온형 🦥 → MBSR
- OXX = 우울형 🐻 → PPT + MBSR
- XOX = 불안형 🐰 → ACT + MBSR
- XXO = 직무스트레스형 🦔 → ACT + CBT
- OOX = 우울+불안형 🦌 → PPT + ACT + CBT
- OXO = 우울+직무스트레스형 🦫 → PPT + ACT
- **XOO = 불안+직무스트레스형 🐿️ → ACT + CBT** (이대리 유형)
- OOO = 위기형 🦊 → PPT + ACT + DBT

#### 3. **2단계 필터링 전략**
- **1차**: 스트레스 유형 → 치료법 매핑
- **2차**: 개인화 (나이, 성별, 직군, 설문키워드)

#### 4. **검색 성능 검증**
```
한국어 쿼리 예시:
- "스트레스 불안 관리" → MBSR 프로그램 연구
- "직무 스트레스 해결방법" → 수용전념치료(ACT)
- "인지행동치료 CBT" → CBT 메타분석 논문
- "마음챙김 명상 MBSR" → MBSR 효과 연구
```

## 💻 API 사용법

### Python에서 사용
```python
from cheeu_rag import quick_healing_message

# 이대리 페르소나 예시
result = quick_healing_message(
    user_input="프로젝트 데드라인에 쫓겨서 밤새 작업하는 날이 많아졌어요. 불안하고 집중이 안 되고 실수도 늘어나서 스트레스가 심해요.",
    nickname="이대리",
    age=27,
    gender="여성",
    occupation="19. 정보통신",
    depression=False,
    anxiety=True,
    work_stress=True,
    survey_keywords=["피로감", "번아웃", "압박감", "집중력 저하", "수면 문제"],
    openai_api_key="sk-your-key"
)

if result["success"]:
    print(result["result"]["healing_message"])
```

## 🔍 검색 전략

### 1차 필터링: 스트레스 유형 → 치료법
```python
# 예: XOO (불안+직무스트레스형)
therapy_methods = ["ACT", "CBT"]  # 자동 매핑
```

### 2차 필터링: 개인화 검색
```python
main_query = user_input + " " + " ".join(therapy_methods)
sub_queries = [
    "20대 여성",              # 인구통계
    "정보통신 IT개발자",       # 직군
    "피로감 번아웃 압박감"     # 설문키워드
]
```

## 📊 데이터 모델

### UserProfile
```python
@dataclass
class UserProfile:
    nickname: str               # 닉네임
    age: int                   # 나이
    gender: str                # 성별
    occupation: str            # 직업
    stress_type: StressType    # 스트레스 유형
    survey_features: List[str] # 설문 키워드
    personal_keywords: List[str] # 개인 키워드
    msi: float = 75.0         # Mental Stress Index
    psi: float = 68.0         # Physical Stress Index
```

### CheeUCapsule (응답)
```python
@dataclass
class CheeUCapsule:
    success: bool              # 성공 여부
    healing_message: str       # 치유 메시지
    character: str            # 캐릭터 이모지
    stress_type: str          # 스트레스 유형
    therapy_methods_used: List[str]  # 사용된 치료법
    sources: List[str]        # 참조 논문
    keywords_used: List[str]  # 활용 키워드
    confidence_score: float   # 신뢰도 점수
    timestamp: str            # 생성 시간
```

## 📝 환경변수 (.env)

```bash
# 필수
OPENAI_API_KEY=sk-your-api-key-here

# 선택 (기본값 있음)
MODEL_NAME=gpt-5-nano
TEMPERATURE=0.7
VECTOR_DB_PATH=./data/vectordb
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 🧪 테스트

```bash
# 단위 테스트
python test_simple.py

# VectorDB 상태 확인
python -c "from cheeu_rag.vectordb import CheeUVectorDB; print(CheeUVectorDB().health_check())"

# 한국어 검색 테스트
python -c "
from cheeu_rag.vectordb import CheeUVectorDB
vectordb = CheeUVectorDB()
docs = vectordb.search_basic('스트레스 불안 관리', k=2)
print(f'검색 결과: {len(docs)}개')
for doc in docs:
    print(f'- {doc.metadata.get(\"title\", \"제목없음\")}')
"
```

## 🔧 기술 스택

| 계층 | 기술 | 역할 |
|------|------|------|
| Frontend | Streamlit | Web UI |
| Backend | Python 3.8+ | Business Logic |
| LLM | OpenAI gpt-5-nano | Text Generation |
| VectorDB | ChromaDB | Document Storage (419개 논문) |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | 한국어 지원 텍스트 벡터화 |
| Framework | LangChain | Pipeline Orchestration |

## ⚡ 성능 지표

- **VectorDB 문서수**: 419개 한국어 연구논문
- **임베딩 차원**: 384차원
- **검색 신뢰도**: 0.517 (이대리 페르소나 기준)
- **한국어 지원**: 우수 (다국어 모델)
- **응답 시간**: ~3-5초 (VectorDB 검색 + LLM 생성)

## 🎯 핵심 특징

### RAG (Retrieval-Augmented Generation)
- **Retrieval**: VectorDB에서 관련 논문 검색 (한국어 최적화)
- **Augmentation**: 검색 결과를 프롬프트에 통합
- **Generation**: gpt-5-nano로 개인화 메시지 생성

### 개인화 전략
1. **스트레스 유형**: 8가지 분류
2. **치료법 매핑**: 유형별 최적 치료법
3. **인구통계**: 연령대, 성별
4. **직군 특화**: 24개 NCS 분류
5. **설문 키워드**: 개인 증상 반영

### 성능 최적화
- **다국어 임베딩**: 한국어 검색 정확도 향상
- **2단계 필터링**: 검색 범위 축소 및 정확도 향상
- **우선순위 가중치 검색**: 70/30 가중치
- **구조화된 프롬프트**: 일관된 출력 형식
- **재시도 로직**: max_retries=3

## 📞 사용 가이드

### 1. 개발 환경에서 실행
```bash
cd CheeU-Final-Release
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env에 OPENAI_API_KEY 설정
streamlit run streamlit_app.py
```

### 2. 프로덕션 배포 시 주의사항
- **API 키 보안**: 환경변수로 관리
- **VectorDB 경로**: 절대경로 설정 권장
- **메모리 사용량**: 임베딩 모델 로딩 시 ~500MB
- **네트워크**: OpenAI API 접근 필요

## 🔄 v1.0 → v2.0 변경사항

### ✅ 개선사항
1. **임베딩 모델 업그레이드**: 한국어 지원 강화
2. **VectorDB 재구성**: 새로운 임베딩으로 전체 재처리
3. **검색 성능 향상**: 한국어 쿼리 정확도 대폭 개선
4. **프롬프트 검증**: 실제 논문 내용 포함 확인

### 🔧 기술적 변경
- `all-MiniLM-L6-v2` → `paraphrase-multilingual-MiniLM-L12-v2`
- VectorDB 차원: 384차원 유지 (호환성)
- 검색 신뢰도: 향상됨
- 모델 지원: gpt-5-nano 추가

---

## 📋 체크리스트

팀원들이 사용하기 전 확인사항:

- [ ] Python 3.8+ 설치됨
- [ ] 가상환경 생성 및 활성화
- [ ] requirements.txt 의존성 설치
- [ ] .env 파일에 OPENAI_API_KEY 설정
- [ ] `python test_simple.py` 실행 성공
- [ ] Streamlit 앱 정상 구동

**🎉 v2.0 최종 버전 - 팀원 배포 준비 완료!**
