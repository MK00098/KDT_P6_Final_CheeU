#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU RAG+LLM Pipeline - Main Controller
모듈화된 통합 API 및 인터페이스
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# CheeU 모듈들 임포트
from .vectordb import CheeUVectorDB
from .chatbot import CheeUChatbot
from .models import (
    StressType, UserProfile, CheeUCapsule,
    STRESS_TYPE_PROFILES, OCCUPATION_KEYWORDS
)


class CheeURagPipeline:
    """
    CheeU RAG+LLM Pipeline - 프로덕션 통합 API
    
    기존 인터페이스와의 호환성을 유지하면서 새로운 모듈 구조를 사용
    CheeU Vector DB와 Chatbot 모듈을 통합하는 메인 인터페이스
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 vector_db_path: Optional[str] = None,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.7,
                 max_retries: int = 3):
        """
        RAG+LLM 파이프라인 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            vector_db_path: CheeU Vector DB 경로 (None시 기본 경로 사용)
            model_name: LLM 모델명
            temperature: 생성 온도
            max_retries: 최대 재시도 횟수
        """
        self.logger = logging.getLogger(__name__)
        
        # API 키 설정
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY를 설정해주세요.")
        
        # 기본 경로 설정
        if vector_db_path is None:
            base_dir = Path(__file__).parent.parent
            vector_db_path = str(base_dir / "data" / "vectordb")
        
        # 모듈들 초기화
        try:
            self.vectordb = CheeUVectorDB(vector_db_path)
            self.chatbot = CheeUChatbot(
                openai_api_key=self.openai_api_key,
                vector_db_path=vector_db_path,
                model_name=model_name,
                temperature=temperature,
                max_retries=max_retries
            )
            
            self.logger.info("✅ CheeU RAG+LLM Pipeline 프로덕션 버전 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ RAG+LLM Pipeline 초기화 실패: {e}")
            raise
    
    # ==================== 메인 API 메서드 ====================
    
    def generate_healing_message(self, 
                                user_input: str,
                                user_profile: UserProfile) -> CheeUCapsule:
        """
        치유 메시지 생성 (메인 API)
        
        Args:
            user_input: 사용자 입력
            user_profile: 사용자 프로필
            
        Returns:
            치유 캡슐 객체
        """
        try:
            # 입력 검증
            if not user_input or not user_input.strip():
                raise ValueError("사용자 입력이 비어있습니다.")
            
            if not isinstance(user_profile, UserProfile):
                raise ValueError("올바른 UserProfile 객체가 필요합니다.")
            
            self.logger.info(f"🎯 치유 메시지 생성 시작: {user_profile.nickname}님")
            
            # Chatbot 모듈에 위임
            result = self.chatbot.generate_healing_message(user_input, user_profile)
            
            self.logger.info(f"✅ 치유 메시지 생성 완료: {user_profile.nickname}님 (성공: {result.success})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 치유 메시지 생성 실패: {e}")
            # 에러 캡슐 반환
            return CheeUCapsule(
                success=False,
                healing_message="시스템 오류로 치유 메시지를 생성할 수 없습니다.",
                character="CheeU 톡톡",
                stress_type="오류",
                therapy_methods_used=[],
                sources=[],
                keywords_used=[],
                confidence_score=0.0,
                timestamp="2024-01-01T00:00:00",
                fallback=True,
                error=str(e)
            )
    
    def generate_healing_capsule_simple(self, 
                                      user_input: str,
                                      user_profile: UserProfile) -> CheeUCapsule:
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
        try:
            return self.vectordb.search_basic(query, k)
        except Exception as e:
            self.logger.error(f"❌ 문서 검색 실패: {e}")
            return []
    
    def search_with_priority(self, 
                           main_query: str,
                           sub_queries: List[str],
                           k: int = 3) -> List:
        """우선순위 기반 문서 검색"""
        try:
            return self.vectordb.search_with_priority_weighting(
                main_query=main_query,
                sub_queries=sub_queries,
                k=k
            )
        except Exception as e:
            self.logger.error(f"❌ 우선순위 검색 실패: {e}")
            return []
    
    def calculate_confidence(self, docs: List, keywords: List[str]) -> float:
        """검색 결과 신뢰도 계산"""
        try:
            return self.vectordb.calculate_search_confidence(docs, keywords)
        except Exception as e:
            self.logger.error(f"❌ 신뢰도 계산 실패: {e}")
            return 0.0
    
    # ==================== 캐릭터 & 유형 관리 ====================
    
    def get_character_info(self, stress_type: StressType) -> Dict[str, Any]:
        """캐릭터 정보 조회"""
        try:
            return self.chatbot.get_character_info(stress_type)
        except Exception as e:
            self.logger.error(f"❌ 캐릭터 정보 조회 실패: {e}")
            return {"error": str(e)}
    
    def get_stress_types(self) -> Dict[str, str]:
        """스트레스 유형 목록 반환"""
        return {stress_type.name: stress_type.value for stress_type in StressType}
    
    def get_occupation_keywords(self, occupation: str) -> List[str]:
        """직군별 키워드 반환"""
        return OCCUPATION_KEYWORDS.get(occupation, [])
    
    def get_available_occupations(self) -> List[str]:
        """사용 가능한 직군 목록 반환"""
        return list(OCCUPATION_KEYWORDS.keys())
    
    # ==================== 시스템 상태 관리 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """시스템 전체 상태 확인 (상세 버전)"""
        try:
            vectordb_status = self.vectordb.health_check()
            chatbot_status = self.chatbot.health_check()
            
            overall_status = vectordb_status and chatbot_status.get("overall_status", False)
            
            result = {
                "vectordb_status": vectordb_status,
                "chatbot_status": chatbot_status.get("overall_status", False),
                "llm_status": chatbot_status.get("llm_status", False),
                "overall_status": overall_status,
                "pipeline_version": "1.0.0-production",
                "timestamp": "2024-01-01T00:00:00",
                "details": {
                    "vectordb": "정상" if vectordb_status else "오류",
                    "chatbot": chatbot_status.get("details", {}),
                    "modules_loaded": ["vectordb", "chatbot", "pipeline"]
                }
            }
            
            if overall_status:
                self.logger.info("✅ RAG+LLM Pipeline 전체 시스템 정상")
            else:
                self.logger.warning("⚠️ RAG+LLM Pipeline 일부 시스템 오류")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 상태 확인 실패: {e}")
            return {
                "vectordb_status": False,
                "chatbot_status": False,
                "llm_status": False,
                "overall_status": False,
                "pipeline_version": "1.0.0-production",
                "timestamp": "2024-01-01T00:00:00",
                "error": str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 조회 (상세 버전)"""
        try:
            vectordb_info = self.vectordb.get_collection_info()
            
            return {
                "pipeline_version": "1.0.0-production",
                "vectordb_info": vectordb_info,
                "stress_types_count": len(STRESS_TYPE_PROFILES),
                "characters_count": len(STRESS_TYPE_PROFILES),
                "occupations_count": len(OCCUPATION_KEYWORDS),
                "api_key_configured": bool(self.openai_api_key),
                "modules": {
                    "vectordb": "CheeUVectorDB",
                    "chatbot": "CheeUChatbot", 
                    "pipeline": "CheeURagPipeline"
                },
                "features": [
                    "우선순위 가중치 검색 (70/30)",
                    "8가지 스트레스 유형 지원",
                    "25개 직군별 키워드 매핑",
                    "폴백 시스템",
                    "신뢰도 점수 계산",
                    "배치 처리 지원"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 정보 조회 실패: {e}")
            return {
                "pipeline_version": "1.0.0-production",
                "error": str(e)
            }
    
    # ==================== 유틸리티 메서드 ====================
    
    def create_user_profile(self,
                          nickname: str,
                          age: int,
                          gender: str,
                          occupation: str,
                          stress_type: StressType,
                          survey_keywords: List[str],
                          msi: float = 75.0,
                          psi: float = 68.0) -> UserProfile:
        """사용자 프로필 생성 헬퍼"""
        try:
            return UserProfile(
                nickname=nickname,
                age=age,
                gender=gender,
                occupation=occupation,
                stress_type=stress_type,
                survey_features=survey_keywords or [],  # 설문 키워드를 설문 특징으로 사용
                personal_keywords=survey_keywords or [],  # 설문 키워드를 개인 키워드로도 사용
                msi=msi,
                psi=psi
            )
        except Exception as e:
            self.logger.error(f"❌ 사용자 프로필 생성 실패: {e}")
            raise ValueError(f"사용자 프로필 생성 실패: {e}")
    
    def determine_stress_type(self, 
                            depression: bool,
                            anxiety: bool,
                            work_stress: bool) -> StressType:
        """스트레스 유형 결정 헬퍼 (개선된 로직)"""
        try:
            code = ""
            code += "O" if depression else "X"
            code += "O" if anxiety else "X"
            code += "O" if work_stress else "X"
            
            stress_type_mapping = {
                "XXX": StressType.XXX,  # 평온형
                "OXX": StressType.OXX,  # 우울형
                "XOX": StressType.XOX,  # 불안형
                "XXO": StressType.XXO,  # 직무스트레스형
                "OOX": StressType.OOX,  # 우울+불안형
                "OXO": StressType.OXO,  # 우울+직무스트레스형
                "XOO": StressType.XOO,  # 불안+직무스트레스형
                "OOO": StressType.OOO   # 위기형
            }
            
            result = stress_type_mapping.get(code, StressType.XXX)
            self.logger.debug(f"💡 스트레스 유형 결정: {code} -> {result.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 스트레스 유형 결정 실패: {e}")
            return StressType.XXX  # 기본값
    
    # ==================== 배치 처리 메서드 ====================
    
    def batch_generate_healing_messages(self, 
                                       requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """여러 치유 메시지 배치 생성 (안정성 강화)"""
        if not requests:
            return []
        
        results = []
        self.logger.info(f"🔄 배치 처리 시작: {len(requests)}개 요청")
        
        for i, request in enumerate(requests):
            try:
                # 요청 검증
                user_input = request.get("user_input", "").strip()
                if not user_input:
                    raise ValueError("user_input이 비어있습니다")
                
                user_profile = request.get("user_profile")
                
                # UserProfile이 없으면 생성
                if not user_profile:
                    user_profile = self.create_user_profile(
                        nickname=request.get("nickname", f"사용자{i+1}"),
                        age=request.get("age", 25),
                        gender=request.get("gender", "기타"),
                        occupation=request.get("occupation", "기타"),
                        stress_type=request.get("stress_type", StressType.XXX),
                        personal_keywords=request.get("personal_keywords", [])
                    )
                
                # 치유 메시지 생성
                healing_capsule = self.generate_healing_message(user_input, user_profile)
                
                results.append({
                    "request_id": i,
                    "success": True,
                    "result": healing_capsule.to_dict()
                })
                
                self.logger.debug(f"✅ 배치 처리 완료 #{i}: {user_profile.nickname}")
                
            except Exception as e:
                results.append({
                    "request_id": i,
                    "success": False,
                    "error": str(e)
                })
                self.logger.error(f"❌ 배치 처리 오류 #{i}: {e}")
        
        success_count = sum(1 for r in results if r["success"])
        self.logger.info(f"🔄 배치 처리 완료: {success_count}/{len(requests)} 성공")
        
        return results
    
    # ==================== 캐시 및 성능 관리 ====================
    
    def clear_cache(self):
        """전체 캐시 클리어"""
        try:
            self.vectordb.clear_cache()
            self.logger.info("🧹 전체 캐시 클리어 완료")
        except Exception as e:
            self.logger.error(f"❌ 캐시 클리어 실패: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        try:
            vectordb_info = self.vectordb.get_collection_info()
            
            return {
                "vectordb_cache_size": vectordb_info.get("cache_size", 0),
                "document_count": vectordb_info.get("document_count", 0),
                "avg_document_length": vectordb_info.get("avg_document_length", 0),
                "embedding_dimension": vectordb_info.get("embedding_dimension", 384),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 성능 통계 조회 실패: {e}")
            return {"error": str(e)}