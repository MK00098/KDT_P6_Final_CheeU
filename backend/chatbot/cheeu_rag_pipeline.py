#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU RAG Pipeline - 모듈화된 통합 API
새로운 모듈 구조를 orchestrate하는 고수준 인터페이스
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# CheeU 모듈들 임포트
from cheeu_vectordb import CheeUVectorDB
from cheeu_chatbot import (
    CheeUChatbot, StressType, UserProfile, HealingCapsule,
    STRESS_TYPE_PROFILES, OCCUPATION_KEYWORDS, TherapyMethod, StressTypeProfile
)

# .env 파일 자동 로드
load_dotenv()


class CheeURagPipeline:
    """
    CheeU RAG Pipeline - 통합 API 인터페이스
    
    기존 인터페이스와의 호환성을 유지하면서 새로운 모듈 구조를 사용
    VectorDB와 Chatbot 모듈을 orchestrate하는 고수준 래퍼
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 vector_db_path: str = "../papers/vectordb",
                 model_name: str = "gpt-4o",
                 temperature: float = 0.7):
        """
        RAG 파이프라인 초기화
        
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
        
        # 모듈들 초기화
        self.vectordb = CheeUVectorDB(vector_db_path)
        self.chatbot = CheeUChatbot(
            openai_api_key=self.openai_api_key,
            vector_db_path=vector_db_path,
            model_name=model_name,
            temperature=temperature
        )
        
        self.logger.info("✅ CheeU RAG Pipeline 초기화 완료")
    
    # ==================== 메인 API 메서드 ====================
    
    def generate_healing_message(self, 
                                user_input: str,
                                user_profile: UserProfile) -> HealingCapsule:
        """
        치유 메시지 생성 (새로운 모듈화된 방식)
        
        Args:
            user_input: 사용자 입력
            user_profile: 사용자 프로필
            
        Returns:
            치유 캡슐 객체
        """
        return self.chatbot.generate_healing_message(user_input, user_profile)
    
    def generate_healing_capsule_simple(self, 
                                      user_input: str,
                                      user_profile: UserProfile) -> HealingCapsule:
        """
        간소화된 치유 캡슐 생성 (기존 인터페이스 호환)
        
        Args:
            user_input: 사용자 입력
            user_profile: 사용자 프로필
            
        Returns:
            치유 캡슐 객체
        """
        return self.generate_healing_message(user_input, user_profile)
    
    # ==================== VectorDB 관련 메서드 ====================
    
    def search_documents(self, query: str, k: int = 3) -> List:
        """기본 문서 검색"""
        return self.vectordb.search_basic(query, k)
    
    def search_with_priority(self, 
                           main_query: str,
                           sub_queries: List[str],
                           k: int = 3) -> List:
        """우선순위 기반 문서 검색"""
        return self.vectordb.search_with_priority_weighting(
            main_query=main_query,
            sub_queries=sub_queries,
            k=k
        )
    
    def calculate_confidence(self, docs: List, keywords: List[str]) -> float:
        """검색 결과 신뢰도 계산"""
        return self.vectordb.calculate_search_confidence(docs, keywords)
    
    # ==================== 캐릭터 & 유형 관리 ====================
    
    def get_character_info(self, stress_type: StressType) -> Dict[str, Any]:
        """캐릭터 정보 조회"""
        return self.chatbot.get_character_info(stress_type)
    
    def get_stress_types(self) -> Dict[str, str]:
        """스트레스 유형 목록 반환"""
        return {stress_type.name: stress_type.value for stress_type in StressType}
    
    def get_occupation_keywords(self, occupation: str) -> List[str]:
        """직군별 키워드 반환"""
        return OCCUPATION_KEYWORDS.get(occupation, [])
    
    # ==================== 시스템 상태 관리 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """시스템 전체 상태 확인"""
        vectordb_status = self.vectordb.health_check()
        chatbot_status = self.chatbot.health_check()
        
        return {
            "vectordb_status": vectordb_status,
            "chatbot_status": chatbot_status["overall_status"],
            "llm_status": chatbot_status["llm_status"],
            "overall_status": vectordb_status and chatbot_status["overall_status"],
            "pipeline_version": "2.0-modular",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 조회"""
        vectordb_info = self.vectordb.get_collection_info()
        
        return {
            "pipeline_version": "2.0-modular",
            "vectordb_info": vectordb_info,
            "stress_types_count": len(STRESS_TYPE_PROFILES),
            "characters_count": len(STRESS_TYPE_PROFILES),  # 각 스트레스 유형마다 캐릭터 1개
            "occupations_count": 25,  # NCS 24개 + 기타
            "api_key_configured": bool(self.openai_api_key),
            "modules": {
                "vectordb": "CheeUVectorDB",
                "chatbot": "CheeUChatbot", 
                "pipeline": "CheeURagPipeline"
            }
        }
    
    # ==================== 유틸리티 메서드 ====================
    
    def create_user_profile(self, 
                          nickname: str,
                          age: int,
                          gender: str,
                          occupation: str,
                          stress_type: StressType,
                          personal_keywords: List[str],
                          msi: float = 75.0,
                          psi: float = 68.0) -> UserProfile:
        """사용자 프로필 생성 헬퍼"""
        return UserProfile(
            nickname=nickname,
            age=age,
            gender=gender,
            occupation=occupation,
            stress_type=stress_type,
            personal_keywords=personal_keywords,
            msi=msi,
            psi=psi
        )
    
    def determine_stress_type(self, 
                            depression: bool,
                            anxiety: bool,
                            work_stress: bool) -> StressType:
        """스트레스 유형 결정 헬퍼"""
        code = ""
        code += "O" if depression else "X"
        code += "O" if anxiety else "X"
        code += "O" if work_stress else "X"
        
        stress_type_mapping = {
            "XXX": StressType.XXX,
            "OXX": StressType.OXX,
            "XOX": StressType.XOX,
            "XXO": StressType.XXO,
            "OOX": StressType.OOX,
            "OXO": StressType.OXO,
            "XOO": StressType.XOO,
            "OOO": StressType.OOO
        }
        
        return stress_type_mapping.get(code, StressType.XXX)
    
    # ==================== 배치 처리 메서드 ====================
    
    def batch_generate_healing_messages(self, 
                                       requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """여러 치유 메시지 배치 생성"""
        results = []
        
        for i, request in enumerate(requests):
            try:
                user_input = request.get("user_input", "")
                user_profile = request.get("user_profile")
                
                if not user_profile:
                    # 프로필 정보로부터 UserProfile 생성
                    user_profile = self.create_user_profile(
                        nickname=request.get("nickname", f"사용자{i+1}"),
                        age=request.get("age", 25),
                        gender=request.get("gender", "기타"),
                        occupation=request.get("occupation", "기타"),
                        stress_type=request.get("stress_type", StressType.XXX),
                        personal_keywords=request.get("personal_keywords", [])
                    )
                
                healing_capsule = self.generate_healing_message(user_input, user_profile)
                results.append({
                    "request_id": i,
                    "success": True,
                    "result": healing_capsule.to_dict()
                })
                
            except Exception as e:
                results.append({
                    "request_id": i,
                    "success": False,
                    "error": str(e)
                })
                self.logger.error(f"배치 처리 오류 #{i}: {e}")
        
        return results
    


# ==================== 모듈 레벨 함수 ====================

def create_pipeline(openai_api_key: Optional[str] = None,
                   vector_db_path: str = "./논문VectorDB") -> CheeURagPipeline:
    """파이프라인 팩토리 함수"""
    return CheeURagPipeline(
        openai_api_key=openai_api_key,
        vector_db_path=vector_db_path
    )


def quick_healing_message(user_input: str,
                         nickname: str = "사용자",
                         age: int = 25,
                         gender: str = "기타",
                         occupation: str = "기타",
                         depression: bool = False,
                         anxiety: bool = False,
                         work_stress: bool = False,
                         personal_keywords: List[str] = None) -> Dict[str, Any]:
    """빠른 치유 메시지 생성 (올인원 함수)"""
    if personal_keywords is None:
        personal_keywords = []
    
    try:
        # 파이프라인 초기화
        pipeline = create_pipeline()
        
        # 스트레스 유형 결정
        stress_type = pipeline.determine_stress_type(depression, anxiety, work_stress)
        
        # 사용자 프로필 생성
        user_profile = pipeline.create_user_profile(
            nickname=nickname,
            age=age,
            gender=gender,
            occupation=occupation,
            stress_type=stress_type,
            personal_keywords=personal_keywords
        )
        
        # 치유 메시지 생성
        healing_capsule = pipeline.generate_healing_message(user_input, user_profile)
        
        return {
            "success": True,
            "result": healing_capsule.to_dict()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 CheeU RAG Pipeline 2.0 (모듈화) 테스트")
    print("=" * 50)
    
    try:
        # 파이프라인 초기화
        pipeline = create_pipeline()
        
        # 시스템 상태 확인
        print("🔍 시스템 상태:")
        status = pipeline.health_check()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n📊 시스템 정보:")
        info = pipeline.get_system_info()
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        # 빠른 테스트
        print("\n💬 빠른 치유 메시지 테스트:")
        result = quick_healing_message(
            user_input="요즘 너무 우울하고 무기력해요",
            nickname="테스트",
            age=28,
            gender="여성",
            occupation="의료진",
            depression=True,
            personal_keywords=["무기력", "피로감"]
        )
        
        if result["success"]:
            healing_result = result["result"]
            print(f"✅ 성공: {healing_result['character']}")
            print(f"💬 메시지: {healing_result['healing_message'][:100]}...")
            print(f"🎯 신뢰도: {healing_result['confidence_score']:.2f}")
        else:
            print(f"❌ 실패: {result['error']}")
            
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
