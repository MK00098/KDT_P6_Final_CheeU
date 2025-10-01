#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU Chatbot Module - Production Version
GPT-4o 기반 개인화된 치유 메시지 생성
"""

import os
import logging
from typing import Optional

# LangChain 컴포넌트
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# CheeU 모듈
from .vectordb import CheeUVectorDB
from .models import (
    UserProfile, CheeUCapsule, StressType,
    STRESS_TYPE_PROFILES, OCCUPATION_KEYWORDS, PERSONA_KEYWORDS
)


class CheeUChatbot:
    """
    CheeU 챗봇 파이프라인 - 프로덕션 버전
    - VectorDB 기반 검색
    - 개인화된 치유 메시지 생성
    - 8가지 캐릭터 대응
    - 프로덕션 안정성 강화
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 vector_db_path: str = None,
                 model_name: str = "gpt-5-nano",
                 temperature: float = 0.7,
                 max_retries: int = 3):
        """
        챗봇 파이프라인 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            vector_db_path: VectorDB 경로
            model_name: LLM 모델명
            temperature: 생성 온도
            max_retries: 최대 재시도 횟수
        """
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        
        # API 키 설정
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY를 설정해주세요.")
        
        # VectorDB 초기화
        self.vectordb = CheeUVectorDB(vector_db_path)
        
        # LLM 초기화 (재시도 로직 포함)
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model=model_name,
            temperature=temperature,
            max_retries=max_retries
        )
        
        # 프롬프트 템플릿
        self._init_prompt_template()
        
        self.logger.info(f"✅ CheeU 챗봇 파이프라인 초기화 완료 (모델: {model_name})")
    
    def _init_prompt_template(self):
        """치유캡슐 프롬프트 템플릿 초기화 (개선된 버전)"""
        self.prompt_template = PromptTemplate(
            input_variables=[
                "nickname", "age", "gender", "occupation", "stress_type",
                "user_input", "vector_context", "survey_features"
            ],
            template="""CheeU 톡톡이 {nickname}님을 위한 맞춤 CheeU 캡슐을 준비했습니다.

💊 사용자 프로필 분석
- {age}세 {gender} {occupation}
- 스트레스 유형: {stress_type}
- 현재 상황: {user_input}
- 설문지 기반 특징: {survey_features}

📚 전문 연구 자료 분석
{vector_context}

🎯 CheeU 캡슐 생성 지침
위 정보를 바탕으로 {nickname}님에게 적합한 개인화된 CheeU 캡슐을 생성해주세요.

💝 CheeU 캡슐 구성 요소:
1. 💊 캡슐 색상: 적용되는 치료기법에 따른 색상 지정
   - 인지행동치료(CBT): 파란색 💙
   - 마음챙김(MBSR): 초록색 💚  
   - 긍정심리치료(PPT): 노란색 💛
   - 수용전념치료(ACT): 주황색 🧡
   - 변증법적행동치료(DBT): 보라색 💜

2. 🎯 핵심 메시지: {nickname}님의 상황에 구체적으로 공감하며 희망을 주는 메시지

3. 🔧 실천 방법: 연구 자료 기반의 구체적이고 실행 가능한 방법 (3가지)

4. ⭐ 격려 문구: 따뜻하고 희망적인 마무리

💬 치유캡슐 내용:"""
        )
    
    def generate_healing_message(self, 
                                user_input: str,
                                user_profile: UserProfile) -> CheeUCapsule:
        """
        치유 메시지 생성 (안정성 강화 버전)
        
        Args:
            user_input: 사용자 입력
            user_profile: 사용자 프로필
            
        Returns:
            치유 캡슐 객체
        """
        try:
            # 1. 입력 검증
            if not user_input or not user_input.strip():
                raise ValueError("사용자 입력이 비어있습니다.")
            
            if not user_profile or not user_profile.nickname:
                raise ValueError("사용자 프로필이 유효하지 않습니다.")
            
            # 2. 우선순위 기반 검색
            main_query = f"{user_input.strip()} {user_profile.stress_type.value}"
            sub_queries = [
                user_profile.get_age_group(),
                user_profile.gender,
                user_profile.occupation,
                " ".join(user_profile.personal_keywords) if user_profile.personal_keywords else ""
            ]
            
            # 직군별 키워드 추가 (안전하게)
            occupation_keywords = OCCUPATION_KEYWORDS.get(user_profile.occupation, [])
            if occupation_keywords:
                sub_queries.extend(occupation_keywords[:3])  # 상위 3개만
            
            # 페르소나별 특화 키워드 추가 (박서현 최적화)
            persona_key = f"{user_profile.nickname}_{user_profile.occupation}"
            if persona_key in PERSONA_KEYWORDS:
                persona_data = PERSONA_KEYWORDS[persona_key]
                # 스트레스, 라이프스타일, 치료 키워드 추가
                for keyword_type in ["stress_keywords", "lifestyle_keywords", "therapy_focus"]:
                    if keyword_type in persona_data:
                        sub_queries.extend(persona_data[keyword_type][:2])  # 각 타입별 상위 2개
            
            # 빈 문자열 필터링
            sub_queries = [q.strip() for q in sub_queries if q and q.strip()]
            
            self.logger.info(f"🔍 검색 시작: {user_profile.nickname}님 ({user_profile.stress_type.value})")
            
            relevant_docs = self.vectordb.search_with_priority_weighting(
                main_query=main_query,
                sub_queries=sub_queries,
                k=3
            )
            
            if not relevant_docs:
                self.logger.warning(f"⚠️ 검색 결과 없음 - 폴백 캡슐 생성: {user_profile.nickname}")
                return self._generate_fallback_capsule(user_profile, user_input)
            
            # 3. 벡터 컨텍스트 구성
            vector_context = self._format_vector_context(relevant_docs)
            
            # 4. 프롬프트 생성 및 LLM 호출 (재시도 로직)
            prompt = self.prompt_template.format(
                nickname=user_profile.nickname,
                age=user_profile.age,
                gender=user_profile.gender,
                occupation=user_profile.occupation,
                stress_type=user_profile.stress_type.value,
                user_input=user_input,
                vector_context=vector_context,
                survey_features=", ".join(user_profile.personal_keywords) if user_profile.personal_keywords else "없음"
            )
            
            self.logger.info(f"🤖 {user_profile.nickname}님의 치유 캡슐 생성 중...")
            
            # LLM 호출 (재시도 포함)
            response = None
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    response = self.llm.predict(prompt)
                    if response and response.strip():
                        break
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"⚠️ LLM 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        raise e
            
            if not response or not response.strip():
                raise ValueError("LLM 응답이 비어있습니다.")
            
            # 5. 캐릭터 및 결과 구성
            stress_profile = STRESS_TYPE_PROFILES[user_profile.stress_type]
            confidence = self.vectordb.calculate_search_confidence(
                relevant_docs, user_profile.personal_keywords or []
            )
            
            healing_capsule = CheeUCapsule(
                success=True,
                healing_message=response.strip(),
                character=f"{stress_profile.emoji} CheeU 톡톡",
                stress_type=user_profile.stress_type.value,
                therapy_methods_used=[m.korean_name for m in stress_profile.therapy_methods],
                sources=[doc.metadata.get('filename', 'Unknown') for doc in relevant_docs],
                keywords_used=user_profile.personal_keywords or [],
                confidence_score=confidence,
                timestamp="2024-01-01T00:00:00",
                fallback=False
            )
            
            self.logger.info(f"✅ 치유 캡슐 생성 완료: {user_profile.nickname}님 (신뢰도: {confidence:.2f})")
            return healing_capsule
            
        except Exception as e:
            self.logger.error(f"❌ 치유 메시지 생성 실패: {e}")
            return self._generate_error_capsule(user_profile, user_input, str(e))
    
    def _format_vector_context(self, docs) -> str:
        """벡터 검색 결과를 컨텍스트로 포맷팅 (개선된 버전)"""
        if not docs:
            return "관련 연구 자료를 찾을 수 없습니다."
        
        context_parts = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            source = doc.metadata.get('filename', f'연구자료{i+1}')
            
            # 내용 길이 제한 (토큰 절약)
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_parts.append(f"[{source}] {content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_fallback_capsule(self, user_profile: UserProfile, user_input: str) -> CheeUCapsule:
        """폴백 치유 캡슐 생성 (개선된 버전)"""
        stress_profile = STRESS_TYPE_PROFILES.get(user_profile.stress_type)
        
        if not stress_profile:
            # 최종 폴백
            return self._generate_error_capsule(user_profile, user_input, "스트레스 유형을 찾을 수 없습니다.")
        
        # 스트레스별 맞춤 폴백 메시지
        fallback_messages = {
            StressType.XXX: "💚 평온 치유캡슐을 전달합니다. 지금 이 순간의 평온함을 느껴보세요. 당신은 충분히 잘하고 있어요.",
            StressType.OXX: "💛 희망 치유캡슐을 전달합니다. 힘든 마음이 느껴집니다. 작은 것부터 시작해보세요. 당신은 혼자가 아니에요.",
            StressType.XOX: "💙 안정 치유캡슐을 전달합니다. 불안한 마음을 이해해요. 깊게 숨을 쉬고 현재에 집중해보세요.",
            StressType.XXO: "🧡 균형 치유캡슐을 전달합니다. 일이 버겁게 느껴지시는군요. 우선순위를 정하고 하나씩 해결해보세요.",
            StressType.OOX: "💜 정리 치유캡슐을 전달합니다. 복잡한 감정들이 얽혀있는 것 같아요. 천천히 정리해나가봐요.",
            StressType.OXO: "🤍 회복 치유캡슐을 전달합니다. 많이 지치셨을 것 같아요. 잠깐 쉬어가도 괜찮습니다.",
            StressType.XOO: "💚 중심 치유캡슐을 전달합니다. 바쁘고 걱정이 많으시군요. 마음챙김으로 중심을 잡아보세요.",
            StressType.OOO: "❤️ 응급 치유캡슐을 전달합니다. 지금 당장 안전이 우선입니다. 주변 도움을 받는 것이 용기입니다."
        }
        
        message = fallback_messages.get(user_profile.stress_type, 
                                       "💝 기본 치유캡슐을 전달합니다. 지금 이 순간, 당신은 충분히 잘하고 있어요.")
        
        return CheeUCapsule(
            success=True,
            healing_message=f"{stress_profile.emoji} {message}\n\n🔧 실천 방법:\n• 현재 감정을 인정하고 받아들이기\n• 깊은 호흡으로 마음 진정하기\n• 작은 성취 경험하기",
            character=f"{stress_profile.emoji} CheeU 톡톡",
            stress_type=user_profile.stress_type.value,
            therapy_methods_used=[m.korean_name for m in stress_profile.therapy_methods],
            sources=[],
            keywords_used=user_profile.personal_keywords or [],
            confidence_score=0.3,
            timestamp="2024-01-01T00:00:00",
            fallback=True
        )
    
    def _generate_error_capsule(self, user_profile: UserProfile, user_input: str, error: str) -> CheeUCapsule:
        """에러 캡슐 생성"""
        return CheeUCapsule(
            success=False,
            healing_message="시스템 오류로 인해 치유 캡슐을 생성할 수 없습니다. 잠시 후 다시 시도해주세요.",
            character="CheeU 톡톡",
            stress_type="오류",
            therapy_methods_used=[],
            sources=[],
            keywords_used=[],
            confidence_score=0.0,
            timestamp="2024-01-01T00:00:00",
            fallback=True,
            error=error
        )
    
    def get_character_info(self, stress_type: StressType) -> dict:
        """캐릭터 정보 조회 (개선된 버전)"""
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
            "therapy_methods": [m.korean_name for m in stress_profile.therapy_methods],
            "method_count": len(stress_profile.therapy_methods)
        }
    
    def health_check(self) -> dict:
        """챗봇 시스템 상태 확인 (강화된 버전)"""
        results = {
            "timestamp": "2024-01-01T00:00:00",
            "vectordb_status": False,
            "llm_status": False,
            "overall_status": False,
            "details": {}
        }
        
        try:
            # 1. VectorDB 상태 확인
            vectordb_status = self.vectordb.health_check()
            results["vectordb_status"] = vectordb_status
            results["details"]["vectordb"] = "정상" if vectordb_status else "오류"
            
            # 2. LLM 연결 테스트
            try:
                test_response = self.llm.predict("안녕하세요", max_tokens=10)
                llm_status = bool(test_response and len(test_response.strip()) > 0)
            except Exception as e:
                llm_status = False
                results["details"]["llm_error"] = str(e)
            
            results["llm_status"] = llm_status
            results["details"]["llm"] = "정상" if llm_status else "오류"
            
            # 3. 전체 상태
            results["overall_status"] = vectordb_status and llm_status
            
            if results["overall_status"]:
                self.logger.info("✅ CheeU 챗봇 시스템 정상 상태")
            else:
                self.logger.warning("⚠️ CheeU 챗봇 시스템 일부 오류")
            
        except Exception as e:
            self.logger.error(f"❌ 챗봇 상태 확인 실패: {e}")
            results["details"]["system_error"] = str(e)
        
        return results