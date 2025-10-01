# 🤖 CheeU LLM 모듈 - AI 치유캡슐 생성 시스템

CheeU 프로젝트의 핵심 AI 모듈로, GPT-4o 기반 개인화 치유캡슐을 생성합니다.

## 🎯 핵심 기능

### 1. 스트레스 유형별 치유캡슐 생성
- **8가지 스트레스 유형** 자동 분류 및 맞춤 치료
- **과학적 근거 기반** 치료기법 적용
- **개인화된 메시지** 생성

### 2. RAG (Retrieval-Augmented Generation) 파이프라인
- **419개 심리학 논문** 청크 기반 컨텍스트 생성
- **우선순위 검색** 알고리즘으로 관련성 높은 자료 추출
- **벡터 유사도 기반** 지식 검색

## 🏗️ 아키텍처

```
사용자 입력 + HRV 데이터
        ↓
스트레스 유형 분류 (8종)
        ↓
VectorDB 검색 (ChromaDB)
        ↓
치료기법 매핑
        ↓
GPT-4o 프롬프트 생성
        ↓
CheeU 톡톡 치유캡슐
```

## 📊 스트레스 유형 분류

| 코드 | 유형 | 이모지 | 주요 치료기법 |
|------|------|--------|---------------|
| **XXX** | 평온형 | 🦥 | MBSR |
| **OXX** | 우울형 | 🐻 | PPT + MBSR |
| **XOX** | 불안형 | 🐰 | MBSR + ACT |
| **XXO** | 직무스트레스형 | 🦔 | ACT + MBSR |
| **OOX** | 우울+불안형 | 🦌 | PPT + ACT |
| **OXO** | 우울+직무스트레스형 | 🦫 | PPT + ACT |
| **XOO** | 불안+직무스트레스형 | 🐿️ | MBSR + ACT |
| **OOO** | 위기형 | 🦊 | ACT + PPT |

## 🧠 치료기법 (Evidence-Based)

### MBSR (마음챙김 기반 스트레스 감소)
- **대상증상**: 스트레스, 불안, 우울
- **핵심기술**: 마음챙김 명상, 자기돌봄, 스트레스 조기 신호 인식
- **근거수준**: Meta-Analysis

### PPT (긍정심리치료)
- **대상증상**: 우울, 무기력, 자존감
- **핵심기술**: 성격강점 활용, 긍정경험 향유, 감사 일기
- **근거수준**: RCT

### ACT (수용전념치료)
- **대상증상**: 불안, 우울, 직무스트레스
- **핵심기술**: 현재순간 인식, 수용, 가치기반 행동
- **근거수준**: Meta-Analysis

## 💻 기술 스택

### LLM & AI
- **OpenAI GPT-4o**: 자연어 생성 엔진
- **Temperature**: 0.7 (창의성과 일관성 균형)
- **LangChain**: RAG 파이프라인 프레임워크

### 벡터 데이터베이스
- **ChromaDB**: 벡터 저장소
- **HuggingFace Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **임베딩 차원**: 384차원
- **데이터**: 419개 심리학 논문 청크

### 검색 알고리즘
- **우선순위 가중치 검색**: 메인쿼리(0.7) + 서브쿼리(0.3)
- **메인쿼리**: 사용자입력 + 스트레스유형
- **서브쿼리**: 연령, 성별, 직군, 개인키워드

## 🚀 사용 방법

### 기본 사용법
```python
from cheeu_rag_pipeline import CheeURagPipeline, StressType, UserProfile

# 파이프라인 초기화
pipeline = CheeURagPipeline()

# 사용자 프로필 생성
user_profile = UserProfile(
    nickname="테스트",
    age=28,
    gender="여성",
    occupation="05. 보건·의료",
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

### 빠른 테스트
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

## 📋 API 레퍼런스

### HealingCapsule 출력 형식
```python
{
    "success": True,
    "healing_message": "CheeU 톡톡이 제공하는 맞춤 치유캡슐...",
    "character": "CheeU 톡톡",
    "stress_type": "우울형",
    "therapy_methods_used": ["긍정심리치료", "마음챙김 기반 스트레스 감소"],
    "confidence_score": 0.85,
    "sources": ["논문1.pdf", "연구자료2.pdf"],
    "timestamp": "2023-09-15T10:30:00"
}
```

## 🔧 설정

### 환경 변수
```bash
# .env 파일
OPENAI_API_KEY=sk-your-api-key-here
```

### 성능 튜닝
- **검색 Top-K**: 3개 (기본값)
- **가중치**: 메인 0.7, 서브 0.3
- **응답 시간**: 2-3초 목표
- **신뢰도 임계값**: 0.6 이상

## 📊 성능 지표

- **벡터 검색 정확도**: 95%+
- **응답 생성 시간**: 2-3초
- **컨텍스트 매칭률**: 90%+
- **치료기법 적용률**: 100%

## 🧪 테스트

### 헬스 체크
```python
pipeline = CheeURagPipeline()
status = pipeline.health_check()
print("시스템 상태:", status['overall_status'])
```

### 시스템 정보
```python
info = pipeline.get_system_info()
print("스트레스 유형 수:", info['stress_types_count'])
print("VectorDB 문서 수:", info['vectordb_info']['document_count'])
```

## 🔍 디버깅

### 일반적인 문제

**1. OpenAI API 키 오류**
```
ValueError: OpenAI API 키가 설정되지 않았습니다.
```
→ `.env` 파일에 `OPENAI_API_KEY` 설정 확인

**2. VectorDB 연결 실패**
```
ChromaDB 컬렉션을 찾을 수 없습니다.
```
→ `./논문VectorDB` 디렉토리 존재 여부 확인

**3. 낮은 신뢰도 점수**
```
confidence_score < 0.6
```
→ `personal_keywords` 관련성 향상 또는 검색 파라미터 조정

## 📈 향후 계획

- [ ] 다국어 지원 (영어, 일본어)
- [ ] 실시간 HRV 연동 강화
- [ ] 치료기법 확장 (DBT, IPT 추가)
- [ ] 성능 최적화 (응답시간 1초 이내)
- [ ] A/B 테스트 프레임워크

---

**CheeU LLM 모듈** - 과학적 근거와 AI 기술로 개인 맞춤형 치유를 제공합니다 🤖💊