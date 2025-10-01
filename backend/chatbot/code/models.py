#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU RAG Pipeline - Data Models
스트레스 유형, 사용자 프로필, 치유캡슐 등 핵심 데이터 모델 정의
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


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
    """사용자 프로필"""
    # 기본 정보
    nickname: str
    age: int
    gender: str  # "남성", "여성", "기타"
    occupation: str
    
    # 스트레스 분석
    stress_type: StressType
    survey_features: List[str]  # 설문지 기반 사용자 특징
    personal_keywords: List[str]  # 개인 키워드 리스트
    
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
class CheeUCapsule:
    """CheeU 캡슐 응답"""
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
    ),
    "CBT": TherapyMethod(
        name="CBT",
        korean_name="인지행동치료",
        target_symptoms=["우울", "불안", "부정적 사고"],
        core_techniques=["인지 재구조화", "행동 활성화", "노출 치료"],
        evidence_level="Meta-Analysis",
        reference="Beck, 1976; CBT 효과성 연구, 2023"
    ),
    "DBT": TherapyMethod(
        name="DBT",
        korean_name="변증법적 행동치료",
        target_symptoms=["감정조절", "대인관계", "위기상황"],
        core_techniques=["마음챙김", "고통 견디기", "감정 조절", "대인관계 효율성"],
        evidence_level="RCT",
        reference="Linehan, 1993; DBT 효과성 연구, 2023"
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
        therapy_methods=[THERAPY_METHODS["ACT"], THERAPY_METHODS["MBSR"]]
    ),
    StressType.XXO: StressTypeProfile(
        emoji="🦔",
        therapy_methods=[THERAPY_METHODS["ACT"], THERAPY_METHODS["CBT"]]
    ),
    StressType.OOX: StressTypeProfile(
        emoji="🦌",
        therapy_methods=[THERAPY_METHODS["PPT"], THERAPY_METHODS["ACT"], THERAPY_METHODS["CBT"]]
    ),
    StressType.OXO: StressTypeProfile(
        emoji="🦫",
        therapy_methods=[THERAPY_METHODS["PPT"], THERAPY_METHODS["ACT"]]
    ),
    StressType.XOO: StressTypeProfile(
        emoji="🐿️",
        therapy_methods=[THERAPY_METHODS["ACT"], THERAPY_METHODS["CBT"]]
    ),
    StressType.OOO: StressTypeProfile(
        emoji="🦊",
        therapy_methods=[THERAPY_METHODS["PPT"], THERAPY_METHODS["ACT"], THERAPY_METHODS["DBT"]]
    )
}

# 직군별 키워드 매핑 (NCS 24개 + 기타)
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
    "19. 정보통신": ["야근", "데드라인", "기술변화", "프로젝트", "버그", "개발스트레스", "코딩", "IT", "스타트업", "멀티태스킹", "불규칙수면", "주의력결핍", "업무과부하", "기술습득압박", "책임감과부하"],
    "20. 기계": ["기계설계", "제조업", "유지보수", "기술개발", "안전관리", "생산성"],
    "21. 금속·재료": ["금속가공", "용접", "안전관리", "품질관리", "재료공학", "제조"],
    "22. 건설": ["건설현장", "안전사고", "프로젝트", "공기단축", "건설관리", "현장작업"],
    "23. 환경·에너지·안전": ["환경보호", "안전관리", "에너지", "환경정책", "안전점검", "위험관리"],
    "24. 인쇄·목재·가구·공예": ["제작기술", "품질관리", "디자인", "수공예", "생산관리", "창작활동"],
    "25. 기타": ["직무스트레스", "업무", "스트레스"],
    
    # 호환성을 위한 간소화 키들
    "경영": ["업무과부하", "관리스트레스", "회의"],
    "의료": ["번아웃", "감정노동", "환자안전", "야근"],
    "교육": ["학습부진", "학부모갈등", "평가스트레스"],
    "IT": ["야근", "데드라인", "기술변화", "프로젝트", "버그"],
    "서비스": ["고객응대", "감정노동", "서비스"],
    "기타": ["직무스트레스", "업무", "스트레스"]
}

# 페르소나별 특화 키워드 매핑
PERSONA_KEYWORDS = {
    "이대리_IT개발자": {
        "stress_keywords": ["불안", "직무스트레스", "수면부족", "주의력결핍", "멀티태스킹", "업무과부하"],
        "lifestyle_keywords": ["야근", "불규칙생활", "커피의존", "기술학습압박"],
        "personality_keywords": ["열정", "책임감", "성장욕구", "완벽주의"],
        "therapy_focus": ["현재순간인식", "수용", "가치기반행동", "인지재구조화", "행동활성화"]
    }
}

# 데모 시나리오 정의
DEMO_SCENARIOS = {
    "이대리_시나리오1": {
        "persona_name": "이대리",
        "user_input": "요즘 프로젝트 데드라인에 쫓겨서 밤새 작업하는 날이 많아졌어요. 불안하고 집중이 안 되고 실수도 늘어나서 스트레스가 심해요.",
        "expected_keywords": ["불안", "데드라인", "야근", "주의력결핍", "스트레스"],
        "expected_therapy": ["ACT", "CBT"],
        "target_response_elements": [
            "현재 순간에 집중하기",
            "완벽주의적 사고 패턴 인식",
            "가치 기반 우선순위 설정",
            "수면 위생 관리"
        ]
    },
    "이대리_시나리오2": {
        "persona_name": "이대리", 
        "user_input": "새로운 기술을 계속 배워야 한다는 압박감과 여러 프로젝트를 동시에 진행하느라 너무 지쳐요. 실수가 늘어나서 더 불안해져요.",
        "expected_keywords": ["기술학습압박", "멀티태스킹", "업무과부하", "불안", "실수"],
        "expected_therapy": ["ACT", "CBT"],
        "target_response_elements": [
            "수용과 전념 기법",
            "인지 왜곡 패턴 수정",
            "업무 우선순위 재정립",
            "자기 효능감 강화"
        ]
    }
}