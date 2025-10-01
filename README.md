# 🤖 CheeU Project - AI 치유캡슐 생성 시스템

GPT-4o 기반 개인화 멘탈 헬스케어 RAG 시스템

## 🎯 시스템 개요

CheeU는 HRV 측정과 심리 설문을 통해 사용자의 스트레스 상태를 분석하고, 
419개의 심리학 논문 청크를 기반으로 개인화된 치유 메시지를 생성하는 RAG 시스템입니다.

### 핵심 특징
- 🧠 **8가지 스트레스 유형** 분류 (평온형~위기형)
- 🤖 **CheeU 톡톡** 통합 AI 상담사
- 📚 **419개 논문 청크** 기반 전문적 조언
- 🤖 **GPT-4o + RAG** 기술로 고품질 응답 생성
- 🏗️ **4개 폴더 구조** - 명확한 책임 분리

## 🚀 주요 기능

### 1. 모듈화된 아키텍처 (NEW!)
- **백엔드**: VectorDB + Chatbot + RAG Pipeline
- **프론트엔드**: Streamlit 웹 인터페이스
- **독립적 모듈**: 각각 독립적으로 테스트 및 배포 가능

### 2. VectorDB (419개 정제된 청크)
- 10개 심리치료 논문에서 추출
- LaTeX, HTML 등 노이즈 제거
- 의미 기반 검색 최적화
- 우선순위 가중치 검색 구현

### 3. RAG Pipeline 2.0
- ChromaDB 벡터 검색
- 메인/서브 조건 우선순위 검색
- GPT-4o 기반 치유 메시지 생성
- 배치 처리 및 헬스 체크 지원

### 4. 8가지 스트레스 타입
```python
XXX: 평온형 🦥
OXX: 우울형 🐻
XOX: 불안형 🐰
XXO: 직무스트레스형 🦔
OOX: 우울+불안형 🦌
OXO: 우울+직무스트레스형 🦫
XOO: 불안+직무스트레스형 🐿️
OOO: 위기형 🦊
```

## 💻 설치 방법

### 1. 필수 요구사항
- Python 3.8+
- OpenAI API Key

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 설정
```bash
# backend/.env 파일 생성
cd backend
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

## 🎮 사용 방법

### 1. 백엔드 모듈 사용법
```python
# backend 폴더에서
import sys
sys.path.append('.')

from cheeu_rag_pipeline import CheeURagPipeline, StressType, UserProfile

# 파이프라인 초기화
pipeline = CheeURagPipeline()

# 사용자 프로필 생성
user_profile = UserProfile(
    nickname="테스트",
    age=28,
    gender="여성",
    occupation="의료진",
    stress_type=StressType.OXX,  # 우울형
    personal_keywords=["무기력", "피로감"]
)

# 치유 메시지 생성
result = pipeline.generate_healing_message(
    user_input="요즘 너무 우울해요",
    user_profile=user_profile
)

print(result.healing_message)
```

### 2. 프론트엔드 실행
```bash
# frontend 폴더에서
cd frontend
streamlit run cheeu_streamlit.py --server.port 8504
```

### 3. 빠른 테스트
```python
from cheeu_rag_pipeline import quick_healing_message

result = quick_healing_message(
    user_input="스트레스가 심해요",
    nickname="테스트",
    age=25,
    depression=True,
    personal_keywords=["불안", "초조"]
)
```

## 🏗️ 아키텍처

### 전체 시스템 구조
```
프론트엔드 (Streamlit)
    ↓ HTTP/API
백엔드 RAG Pipeline 2.0
    ↓
┌─────────────┬─────────────┐
│ VectorDB    │ Chatbot     │
│ 모듈        │ 모듈        │
└─────────────┴─────────────┘
    ↓
ChromaDB (419개 청크)
```

### RAG 파이프라인 흐름
```
사용자 입력 + 프로필
    ↓
스트레스 유형 분류
    ↓
우선순위 검색 (메인: 입력+유형, 서브: 인구통계)
    ↓
VectorDB 검색 (Top-K=3)
    ↓
컨텍스트 구성 + 캐릭터 선택
    ↓
GPT-4o 생성
    ↓
치유캡슐 메시지
```

## 📁 프로젝트 구조

```
CheeU Project/
├── papers/           # 📚 논문 관련
│   ├── raw/         # PDF 논문 저장소
│   ├── vectordb/    # 벡터 데이터베이스 (419개 청크)
│   └── add_papers.py # 논문 추가 스크립트
│
├── backend/         # 🔧 백엔드 API
│   ├── cheeu_chatbot.py      # 치유캡슐 생성
│   ├── cheeu_vectordb.py     # 벡터 DB 관리
│   ├── cheeu_rag_pipeline.py # RAG 파이프라인
│   └── .env                  # 환경 변수
│
├── frontend/        # 🎨 프론트엔드 UI
│   └── cheeu_streamlit.py    # Streamlit 웹앱
│
├── docs/           # 📖 문서
│   ├── LLM_README.md         # LLM 모듈 문서
│   └── requirements.txt      # 의존성
│
└── README.md       # 프로젝트 개요
```

## 📊 성능 지표

- **모듈 로딩**: < 1초
- **검색 속도**: < 0.5초
- **생성 시간**: 2-3초
- **우선순위 검색 정확도**: 95%+
- **사용자 만족도 목표**: 4.0/5.0

## 🔧 API 문서

### 백엔드 모듈

#### `CheeURagPipeline`
통합 RAG Pipeline 2.0 - 모든 모듈을 orchestrate하는 고수준 API

```python
# 초기화
pipeline = CheeURagPipeline(
    openai_api_key="sk-...",
    vector_db_path="./논문VectorDB",
    model_name="gpt-4o",
    temperature=0.7
)

# 메인 API
healing_capsule = pipeline.generate_healing_message(user_input, user_profile)

# 유틸리티
user_profile = pipeline.create_user_profile(...)
stress_type = pipeline.determine_stress_type(depression, anxiety, work_stress)
system_info = pipeline.health_check()
```

#### `UserProfile` 데이터 클래스
```python
@dataclass
class UserProfile:
    nickname: str
    age: int
    gender: str
    occupation: str
    stress_type: StressType
    personal_keywords: List[str]
    msi: float = 75.0  # Mental Stress Index
    psi: float = 68.0  # Physical Stress Index
```

#### `HealingCapsule` 응답 형식
```python
@dataclass
class HealingCapsule:
    healing_message: str
    character: str
    character_emoji: str
    stress_type: str
    confidence_score: float
    sources: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]
```

### 프론트엔드

Streamlit 웹 인터페이스로 다음 기능 제공:
- 📝 사용자 정보 입력 폼
- 💬 실시간 치유 메시지 생성
- 📊 시스템 상태 모니터링

## 📧 실행 방법

### 로컬 개발
```bash
# 1. 백엔드 테스트
cd backend
python cheeu_rag_pipeline.py

# 2. 프론트엔드 실행
cd frontend
streamlit run cheeu_streamlit.py --server.port 8504
```

### 배포 준비
- 백엔드: FastAPI 서버로 확장 가능
- 프론트엔드: Streamlit Cloud 배포 가능
- VectorDB: 클라우드 저장소 마이그레이션 가능

---

**CheeU 치유캡슐 톡 2.0** - 모듈화된 AI 동반자가 당신의 마음 건강을 돌봅니다 💊💬🚀