#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU Chatbot Pipeline Module
치유 캡슐 생성 및 대화 로직
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

# LangChain 컴포넌트
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# CheeU 모듈
from cheeu_vectordb import CheeUVectorDB

# .env 파일 자동 로드
load_dotenv()


class StressType(Enum):
    """8가지 스트레스 유형 - 논문 기반 분류"""
    XXX = "평온형"        # 🦥 평온한 나무늘보형
    OXX = "우울형"        # 🐻 겨울잠 자는 곰형  
    XOX = "불안형"        # 🐰 조급한 토끼형
    XXO = "직무스트레스형"  # 🦔 가시돋친 고슴도치형
    OOX = "우울+불안형"    # 🦌 고뇌하는 사슴형
    OXO = "우울+직무스트레스형"  # 🦫 지친 비버형
    XOO = "불안+직무스트레스형"  # 🐿️ 바쁜 다람쥐형
    OOO = "위기형"        # 🦊 혼란스런 여우형


@dataclass
class TherapyMethod:
    """치료법 정보"""
    name: str
    korean_name: str
    target_symptoms: List[str]
    core_techniques: List[str]
    evidence_level: str  # "RCT", "Meta-Analysis", "Clinical"
    reference: str


@dataclass
class StressTypeProfile:
    """스트레스 유형별 치료 프로필"""
    emoji: str
    therapy_methods: List[TherapyMethod]


@dataclass
class UserProfile:
    """단순화된 사용자 프로필"""
    # 기본 정보
    nickname: str
    age: int
    gender: str  # "남성", "여성", "기타"
    occupation: str
    
    # 스트레스 분석
    stress_type: StressType
    personal_keywords: List[str]  # 설문지에서 추출한 키워드들
    
    # HRV 단순화
    msi: float  # Mental Stress Index
    psi: float  # Physical Stress Index
    
    # 설문 점수 (선택사항)
    phq9_score: Optional[int] = None
    gad7_score: Optional[int] = None  
    koss24_score: Optional[int] = None
    
    def get_age_group(self) -> str:
        """연령대 반환"""
        if self.age < 20:
            return "10대"
        elif self.age < 30:
            return "20대"
        elif self.age < 40:
            return "30대"
        elif self.age < 50:
            return "40대"
        elif self.age < 60:
            return "50대"
        else:
            return "60대 이상"


@dataclass
class HealingCapsule:
    """치유 캡슐 응답"""
    success: bool
    healing_message: str
    character: str
    stress_type: str
    therapy_methods_used: List[str]
    sources: List[str]
    keywords_used: List[str]
    confidence_score: float
    timestamp: str
    fallback: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "success": self.success,
            "healing_message": self.healing_message,
            "character": self.character,
            "stress_type": self.stress_type,
            "therapy_methods_used": self.therapy_methods_used,
            "sources": self.sources,
            "keywords_used": self.keywords_used,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp,
            "fallback": self.fallback,
            "error": self.error
        }


# 논문 기반 치료법 정의
THERAPY_METHODS = {
    "MBSR": TherapyMethod(
        name="MBSR",
        korean_name="마음챙김 기반 스트레스 감소",
        target_symptoms=["스트레스", "불안", "우울"],
        core_techniques=["마음챙김 명상", "자기돌봄", "스트레스 조기 신호 인식"],
        evidence_level="Meta-Analysis",
        reference="Hayes, 1994; K-MBSR 연구, 2023"
    ),
    "PPT": TherapyMethod(
        name="PPT",
        korean_name="긍정심리치료",
        target_symptoms=["우울", "무기력", "자존감"],
        core_techniques=["성격강점 활용", "긍정경험 향유", "감사 일기"],
        evidence_level="RCT",
        reference="Seligman, 2005; 긍정심리 논문, 2023"
    ),
    "ACT": TherapyMethod(
        name="ACT",
        korean_name="수용전념치료",
        target_symptoms=["불안", "우울", "직무스트레스"],
        core_techniques=["현재순간 인식", "수용", "가치기반 행동"],
        evidence_level="Meta-Analysis",
        reference="Hayes, 2006; ACT 효과성 연구, 2023"
    )
}

# 스트레스 유형별 치료 프로필 정의
STRESS_TYPE_PROFILES = {
    StressType.XXX: StressTypeProfile(
        emoji="🦥",
        therapy_methods=[THERAPY_METHODS["MBSR"]]
    ),
    StressType.OXX: StressTypeProfile(
        emoji="🐻",
        therapy_methods=[THERAPY_METHODS["PPT"], THERAPY_METHODS["MBSR"]]
    ),
    StressType.XOX: StressTypeProfile(
        emoji="🐰",
        therapy_methods=[THERAPY_METHODS["MBSR"], THERAPY_METHODS["ACT"]]
    ),
    StressType.XXO: StressTypeProfile(
        emoji="🦔",
        therapy_methods=[THERAPY_METHODS["ACT"], THERAPY_METHODS["MBSR"]]
    ),
    StressType.OOX: StressTypeProfile(
        emoji="🦌",
        therapy_methods=[THERAPY_METHODS["PPT"], THERAPY_METHODS["ACT"]]
    ),
    StressType.OXO: StressTypeProfile(
        emoji="🦫",
        therapy_methods=[THERAPY_METHODS["PPT"], THERAPY_METHODS["ACT"]]
    ),
    StressType.XOO: StressTypeProfile(
        emoji="🐿️",
        therapy_methods=[THERAPY_METHODS["MBSR"], THERAPY_METHODS["ACT"]]
    ),
    StressType.OOO: StressTypeProfile(
        emoji="🦊",
        therapy_methods=[THERAPY_METHODS["ACT"], THERAPY_METHODS["PPT"]]
    )
}

# NCS 기준 24개 직군별 키워드 매핑
OCCUPATION_KEYWORDS = {
    "01. 경영·회계·사무": ["업무과부하", "회계처리", "사무업무", "관리스트레스", "보고서작성", "회의"],
    "02. 금융·보험": ["리스크관리", "고객상담", "실적압박", "규제준수", "금융상품", "투자압박"],
    "03. 교육·자연과학·사회과학": ["학습부진", "학부모갈등", "업무과부하", "평가스트레스", "행정업무", "학생지도", "연구"],
    "04. 법률·경찰·소방·교도·국방": ["치안유지", "법률해석", "공공안전", "위험상황", "순찰", "사법업무"],
    "05. 보건·의료": ["번아웃", "감정노동", "환자안전", "야근", "의료사고", "업무과부하", "응급상황"],
    "06. 사회복지·종교": ["감정노동", "상담업무", "복지서비스", "봉사활동", "사례관리", "클라이언트"],
    "07. 문화·예술·디자인·방송": ["창작스트레스", "작품활동", "경제적불안", "작품평가", "창작슬럼프", "시청률"],
    "08. 운동": ["체력관리", "경기스트레스", "부상위험", "성과압박", "훈련", "시합"],
    "09. 여행·레저": ["고객서비스", "성수기", "관광안내", "레저활동", "계절업무", "서비스"],
    "10. 숙박·음식": ["고객응대", "서비스", "주방업무", "위생관리", "주말근무", "감정노동"],
    "11. 미용·예식": ["고객만족", "서비스업", "미용기술", "예식준비", "감정노동", "트렌드"],
    "12. 비서·사무보조": ["업무지원", "스케줄관리", "사무업무", "업무보조", "문서작업", "일정관리"],
    "13. 농림어업": ["날씨영향", "계절성", "농작물관리", "어업활동", "환경변화", "수확"],
    "14. 식품가공": ["위생관리", "품질관리", "생산라인", "식품안전", "제조업무", "공정관리"],
    "15. 섬유·의복": ["패션트렌드", "제조업무", "품질관리", "의복제작", "소재관리", "디자인"],
    "16. 재료": ["품질관리", "소재개발", "재료과학", "제조공정", "기술개발", "연구개발"],
    "17. 화학": ["화학물질", "안전관리", "실험", "연구개발", "품질관리", "환경관리"],
    "18. 전기·전자": ["회로설계", "전자기기", "기술개발", "품질관리", "전기안전", "유지보수"],
    "19. 정보통신": ["야근", "데드라인", "기술변화", "프로젝트", "버그", "개발스트레스", "코딩", "IT"],
    "20. 기계": ["기계설계", "제조업", "유지보수", "기술개발", "안전관리", "생산성"],
    "21. 금속·재료": ["금속가공", "용접", "안전관리", "품질관리", "재료공학", "제조"],
    "22. 건설": ["건설현장", "안전사고", "프로젝트", "공기단축", "건설관리", "현장작업"],
    "23. 환경·에너지·안전": ["환경보호", "안전관리", "에너지", "환경정책", "안전점검", "위험관리"],
    "24. 인쇄·목재·가구·공예": ["제작기술", "품질관리", "디자인", "수공예", "생산관리", "창작활동"],
    "25. 기타": ["직무스트레스", "업무", "스트레스"],
    # 호환성을 위한 원본 키들 (넘버링 없는 버전)
    "경영·회계·사무": ["업무과부하", "회계처리", "사무업무", "관리스트레스", "보고서작성", "회의"],
    "금융·보험": ["리스크관리", "고객상담", "실적압박", "규제준수", "금융상품", "투자압박"],
    "교육·자연과학·사회과학": ["학습부진", "학부모갈등", "업무과부하", "평가스트레스", "행정업무", "학생지도", "연구"],
    "법률·경찰·소방·교도·국방": ["치안유지", "법률해석", "공공안전", "위험상황", "순찰", "사법업무"],
    "보건·의료": ["번아웃", "감정노동", "환자안전", "야근", "의료사고", "업무과부하", "응급상황"],
    "사회복지·종교": ["감정노동", "상담업무", "복지서비스", "봉사활동", "사례관리", "클라이언트"],
    "문화·예술·디자인·방송": ["창작스트레스", "작품활동", "경제적불안", "작품평가", "창작슬럼프", "시청률"],
    "운동": ["체력관리", "경기스트레스", "부상위험", "성과압박", "훈련", "시합"],
    "여행·레저": ["고객서비스", "성수기", "관광안내", "레저활동", "계절업무", "서비스"],
    "숙박·음식": ["고객응대", "서비스", "주방업무", "위생관리", "주말근무", "감정노동"],
    "미용·예식": ["고객만족", "서비스업", "미용기술", "예식준비", "감정노동", "트렌드"],
    "비서·사무보조": ["업무지원", "스케줄관리", "사무업무", "업무보조", "문서작업", "일정관리"],
    "농림어업": ["날씨영향", "계절성", "농작물관리", "어업활동", "환경변화", "수확"],
    "식품가공": ["위생관리", "품질관리", "생산라인", "식품안전", "제조업무", "공정관리"],
    "섬유·의복": ["패션트렌드", "제조업무", "품질관리", "의복제작", "소재관리", "디자인"],
    "재료": ["품질관리", "소재개발", "재료과학", "제조공정", "기술개발", "연구개발"],
    "화학": ["화학물질", "안전관리", "실험", "연구개발", "품질관리", "환경관리"],
    "전기·전자": ["회로설계", "전자기기", "기술개발", "품질관리", "전기안전", "유지보수"],
    "정보통신": ["야근", "데드라인", "기술변화", "프로젝트", "버그", "개발스트레스", "코딩", "IT"],
    "기계": ["기계설계", "제조업", "유지보수", "기술개발", "안전관리", "생산성"],
    "금속·재료": ["금속가공", "용접", "안전관리", "품질관리", "재료공학", "제조"],
    "건설": ["건설현장", "안전사고", "프로젝트", "공기단축", "건설관리", "현장작업"],
    "환경·에너지·안전": ["환경보호", "안전관리", "에너지", "환경정책", "안전점검", "위험관리"],
    "인쇄·목재·가구·공예": ["제작기술", "품질관리", "디자인", "수공예", "생산관리", "창작활동"],
    "기타": ["직무스트레스", "업무", "스트레스"]
}


class CheeUChatbot:
    """
    CheeU 챗봇 파이프라인
    - VectorDB 기반 검색
    - 개인화된 치유 메시지 생성
    - 8가지 캐릭터 대응
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 vector_db_path: str = "./논문VectorDB",
                 model_name: str = "gpt-4o",
                 temperature: float = 0.7):
        """
        챗봇 파이프라인 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            vector_db_path: VectorDB 경로
            model_name: LLM 모델명
            temperature: 생성 온도
        """
        self.logger = logging.getLogger(__name__)
        
        # API 키 설정
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        # VectorDB 초기화
        self.vectordb = CheeUVectorDB(vector_db_path)
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        # 프롬프트 템플릿
        self._init_prompt_template()
        
        self.logger.info("✅ CheeU 챗봇 파이프라인 초기화 완료")
    
    def _init_prompt_template(self):
        """치유캡슐 제공 방식 프롬프트 템플릿 초기화"""
        self.prompt_template = PromptTemplate(
            input_variables=[
                "nickname", "age", "gender", "occupation", "stress_type",
                "user_input", "vector_context", "personal_keywords"
            ],
            template="""CheeU 톡톡이 {nickname}님을 위한 맞춤 치유캡슐을 준비했습니다.

💊 사용자 프로필 분석
- {age}세 {gender} {occupation}
- 스트레스 유형: {stress_type}
- 현재 상황: {user_input}
- 개인 키워드: {personal_keywords}

📚 전문 연구 자료 분석
{vector_context}

🎯 치유캡슐 생성 지침
위 정보를 바탕으로 {nickname}님에게 적합한 치유캡슐을 생성해주세요.

💝 치유캡슐 구성 요소:
1. 💊 캡슐 색상: 적용되는 치료기법에 따른 색상 지정 (인지행동치료=파란색, 마음챙김=초록색, 긍정심리치료=노란색, 대인관계치료=보라색 등)
2. 🎯 핵심 메시지: {nickname}님의 상황에 구체적으로 공감하며 희망을 주는 메시지
3. 🔧 실천 방법: 연구 자료 기반의 구체적이고 실행 가능한 방법
4. ⭐ 격려 문구: 따뜻하고 희망적인 마무리

💬 치유캡슐 내용:"""
        )
    
    def generate_healing_message(self, 
                                user_input: str,
                                user_profile: UserProfile) -> HealingCapsule:
        """
        치유 메시지 생성 (새로운 간소화 버전)
        
        Args:
            user_input: 사용자 입력
            user_profile: 사용자 프로필
            
        Returns:
            치유 캡슐 객체
        """
        try:
            # 1. 우선순위 기반 검색
            main_query = f"{user_input} {user_profile.stress_type.value}"
            sub_queries = [
                user_profile.get_age_group(),
                user_profile.gender,
                user_profile.occupation,
                " ".join(user_profile.personal_keywords)
            ]
            
            # 직군별 키워드 추가
            occupation_keywords = OCCUPATION_KEYWORDS.get(user_profile.occupation, [])
            sub_queries.extend(occupation_keywords[:3])  # 상위 3개만
            
            relevant_docs = self.vectordb.search_with_priority_weighting(
                main_query=main_query,
                sub_queries=sub_queries,
                k=3
            )
            
            if not relevant_docs:
                return self._generate_fallback_capsule(user_profile)
            
            # 2. 벡터 컨텍스트 구성
            vector_context = self._format_vector_context(relevant_docs)
            
            # 3. 프롬프트 생성 및 LLM 호출
            prompt = self.prompt_template.format(
                nickname=user_profile.nickname,
                age=user_profile.age,
                gender=user_profile.gender,
                occupation=user_profile.occupation,
                stress_type=user_profile.stress_type.value,
                user_input=user_input,
                vector_context=vector_context,
                personal_keywords=", ".join(user_profile.personal_keywords)
            )
            
            self.logger.info(f"🤖 {user_profile.nickname}님의 치유 캡슐 생성 중...")
            response = self.llm.predict(prompt)
            
            # 4. 캐릭터 및 결과 구성
            stress_profile = STRESS_TYPE_PROFILES[user_profile.stress_type]
            confidence = self.vectordb.calculate_search_confidence(
                relevant_docs, user_profile.personal_keywords
            )
            
            return HealingCapsule(
                success=True,
                healing_message=response.strip(),
                character="CheeU 톡톡",
                stress_type=user_profile.stress_type.value,
                therapy_methods_used=[m.korean_name for m in stress_profile.therapy_methods],
                sources=[doc.metadata.get('filename', 'Unknown') for doc in relevant_docs],
                keywords_used=user_profile.personal_keywords,
                confidence_score=confidence,
                timestamp=datetime.now().isoformat(),
                fallback=False
            )
            
        except Exception as e:
            self.logger.error(f"❌ 치유 캡슐 생성 오류: {e}")
            return HealingCapsule(
                success=False,
                healing_message="잠시 후 다시 시도해주세요. 🌟",
                character="시스템",
                stress_type="오류",
                therapy_methods_used=[],
                sources=[],
                keywords_used=[],
                confidence_score=0.0,
                timestamp=datetime.now().isoformat(),
                fallback=True,
                error=str(e)
            )
    
    def _format_vector_context(self, docs) -> str:
        """벡터 검색 결과를 컨텍스트로 포맷팅"""
        if not docs:
            return "관련 연구 자료를 찾을 수 없습니다."
        
        context_parts = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            source = doc.metadata.get('filename', f'연구자료{i+1}')
            context_parts.append(f"[{source}] {content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_fallback_capsule(self, user_profile: UserProfile) -> HealingCapsule:
        """폴백 치유 캡슐 생성"""
        stress_profile = STRESS_TYPE_PROFILES.get(user_profile.stress_type)
        
        fallback_messages = {
            StressType.XXX: "CheeU 톡톡이 평온 치유캡슐 💚을 전달합니다. 지금 이 순간의 평온함을 느껴보세요. 당신은 충분히 잘하고 있어요.",
            StressType.OXX: "CheeU 톡톡이 희망 치유캡슐 💛을 전달합니다. 힘든 마음이 느껴집니다. 작은 것부터 시작해보세요. 당신은 혼자가 아니에요.",
            StressType.XOX: "CheeU 톡톡이 안정 치유캡슐 💙을 전달합니다. 불안한 마음을 이해해요. 깊게 숨을 쉬고 현재에 집중해보세요.",
            StressType.XXO: "CheeU 톡톡이 균형 치유캡슐 🧡을 전달합니다. 일이 버겁게 느껴지시는군요. 우선순위를 정하고 하나씩 해결해보세요.",
            StressType.OOX: "CheeU 톡톡이 정리 치유캡슐 💜을 전달합니다. 복잡한 감정들이 얽혀있는 것 같아요. 천천히 정리해나가봐요.",
            StressType.OXO: "CheeU 톡톡이 회복 치유캡슐 🤍을 전달합니다. 많이 지치셨을 것 같아요. 잠깐 쉬어가도 괜찮습니다.",
            StressType.XOO: "CheeU 톡톡이 중심 치유캡슐 💚을 전달합니다. 바쁘고 걱정이 많으시군요. 마음챙김으로 중심을 잡아보세요.",
            StressType.OOO: "CheeU 톡톡이 응급 치유캡슐 ❤️을 전달합니다. 지금 당장 안전이 우선입니다. 주변 도움을 받는 것이 용기입니다."
        }
        
        message = fallback_messages.get(user_profile.stress_type, 
                                       "CheeU 톡톡이 기본 치유캡슐 💝을 전달합니다. 지금 이 순간, 당신은 충분히 잘하고 있어요.")
        
        return HealingCapsule(
            success=True,
            healing_message=f"{stress_profile.emoji} {message}",
            character="CheeU 톡톡",
            stress_type=user_profile.stress_type.value,
            therapy_methods_used=[m.korean_name for m in stress_profile.therapy_methods],
            sources=[],
            keywords_used=user_profile.personal_keywords,
            confidence_score=0.3,
            timestamp=datetime.now().isoformat(),
            fallback=True
        )
    
    def get_character_info(self, stress_type: StressType) -> Dict[str, Any]:
        """캐릭터 정보 조회"""
        stress_profile = STRESS_TYPE_PROFILES.get(stress_type)
        if not stress_profile:
            return {"error": "스트레스 유형을 찾을 수 없습니다."}
        
        # 스트레스 유형별 톤 정의
        tone_mapping = {
            StressType.XXX: "차분하고 평온한",
            StressType.OXX: "따뜻하고 희망적인", 
            StressType.XOX: "안정적이고 차분한",
            StressType.XXO: "균형잡히고 실용적인",
            StressType.OOX: "공감적이고 위로하는",
            StressType.OXO: "회복에 초점을 둔",
            StressType.XOO: "중심을 잡아주는",
            StressType.OOO: "응급하고 즉각적인"
        }
        
        return {
            "emoji": stress_profile.emoji,
            "name": f"{stress_profile.emoji} CheeU 톡톡",
            "tone": tone_mapping.get(stress_type, "친근하고 도움이 되는"),
            "stress_type": stress_type.value,
            "therapy_methods": [m.korean_name for m in stress_profile.therapy_methods]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """챗봇 시스템 상태 확인"""
        vectordb_status = self.vectordb.health_check()
        
        try:
            # LLM 연결 테스트
            test_response = self.llm.predict("안녕하세요")
            llm_status = len(test_response) > 0
        except Exception as e:
            llm_status = False
            self.logger.error(f"LLM 상태 확인 실패: {e}")
        
        return {
            "vectordb_status": vectordb_status,
            "llm_status": llm_status,
            "overall_status": vectordb_status and llm_status,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 챗봇 초기화
    chatbot = CheeUChatbot()
    
    # 상태 확인
    print("🤖 챗봇 상태 확인:")
    status = chatbot.health_check()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 테스트 사용자 프로필
    test_profile = UserProfile(
        nickname="테스트",
        age=28,
        gender="여성",
        occupation="의료진",
        stress_type=StressType.OXX,
        personal_keywords=["무기력", "피로감"],
        msi=75.0,
        psi=68.0
    )
    
    # 치유 메시지 생성 테스트
    print(f"\n💬 치유 메시지 생성 테스트:")
    healing_capsule = chatbot.generate_healing_message(
        user_input="요즘 너무 피곤하고 우울해요",
        user_profile=test_profile
    )
    
    result = healing_capsule.to_dict()
    print(f"✅ 성공: {result['success']}")
    print(f"🐻 캐릭터: {result['character']}")
    print(f"💬 메시지: {result['healing_message'][:100]}...")
    print(f"🎯 신뢰도: {result['confidence_score']:.2f}")