#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU RAG+LLM Pipeline - API Functions
간편한 사용을 위한 API 함수들
"""

import logging
from typing import List, Dict, Any, Optional

from .pipeline import CheeURagPipeline
from .models import StressType, UserProfile


# 글로벌 파이프라인 인스턴스 (싱글톤 패턴)
_pipeline_instance = None


def create_pipeline(openai_api_key: Optional[str] = None,
                   vector_db_path: Optional[str] = None,
                   model_name: str = "gpt-4o",
                   temperature: float = 0.7) -> CheeURagPipeline:
    """
    파이프라인 생성 함수
    
    Args:
        openai_api_key: OpenAI API 키
        vector_db_path: VectorDB 경로
        model_name: LLM 모델명
        temperature: 생성 온도
        
    Returns:
        초기화된 CheeURagPipeline 인스턴스
    """
    return CheeURagPipeline(
        openai_api_key=openai_api_key,
        vector_db_path=vector_db_path,
        model_name=model_name,
        temperature=temperature
    )


def get_pipeline(openai_api_key: Optional[str] = None,
                vector_db_path: Optional[str] = None) -> CheeURagPipeline:
    """
    싱글톤 파이프라인 인스턴스 반환
    
    Args:
        openai_api_key: OpenAI API 키 (첫 생성시만 사용)
        vector_db_path: VectorDB 경로 (첫 생성시만 사용)
        
    Returns:
        글로벌 파이프라인 인스턴스
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = create_pipeline(
            openai_api_key=openai_api_key,
            vector_db_path=vector_db_path
        )
    
    return _pipeline_instance


def quick_healing_message(user_input: str,
                         nickname: str = "사용자",
                         age: int = 25,
                         gender: str = "기타",
                         occupation: str = "기타",
                         depression: bool = False,
                         anxiety: bool = False,
                         work_stress: bool = False,
                         survey_keywords: Optional[List[str]] = None,
                         openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    빠른 치유 메시지 생성 (올인원 함수)
    
    Args:
        user_input: 사용자 입력
        nickname: 닉네임
        age: 나이
        gender: 성별 ("남성", "여성", "기타")
        occupation: 직업
        depression: 우울 여부
        anxiety: 불안 여부
        work_stress: 직무스트레스 여부
        survey_keywords: 설문 키워드 리스트
        openai_api_key: OpenAI API 키
        
    Returns:
        치유 메시지 결과 딕셔너리
    """
    if survey_keywords is None:
        survey_keywords = []

    try:
        # 파이프라인 초기화
        pipeline = get_pipeline(openai_api_key=openai_api_key)

        # 스트레스 유형 결정
        stress_type = pipeline.determine_stress_type(depression, anxiety, work_stress)

        # 사용자 프로필 생성
        user_profile = pipeline.create_user_profile(
            nickname=nickname,
            age=age,
            gender=gender,
            occupation=occupation,
            stress_type=stress_type,
            survey_keywords=survey_keywords
        )
        
        # 치유 메시지 생성
        healing_capsule = pipeline.generate_healing_message(user_input, user_profile)
        
        return {
            "success": True,
            "result": healing_capsule.to_dict(),
            "user_profile": {
                "nickname": nickname,
                "stress_type": stress_type.value,
                "age_group": user_profile.get_age_group()
            }
        }
        
    except Exception as e:
        logging.error(f"❌ 빠른 치유 메시지 생성 실패: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def batch_healing_messages(requests: List[Dict[str, Any]],
                          openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    여러 치유 메시지 배치 생성
    
    Args:
        requests: 요청 리스트 (각 요청은 quick_healing_message의 파라미터 포함)
        openai_api_key: OpenAI API 키
        
    Returns:
        배치 처리 결과
    """
    try:
        # 파이프라인 초기화
        pipeline = get_pipeline(openai_api_key=openai_api_key)
        
        # 배치 처리
        results = pipeline.batch_generate_healing_messages(requests)
        
        success_count = sum(1 for r in results if r["success"])
        
        return {
            "success": True,
            "total_requests": len(requests),
            "successful_requests": success_count,
            "failed_requests": len(requests) - success_count,
            "results": results
        }
        
    except Exception as e:
        logging.error(f"❌ 배치 치유 메시지 생성 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


def get_stress_type_info(depression: bool = False,
                        anxiety: bool = False,
                        work_stress: bool = False) -> Dict[str, Any]:
    """
    스트레스 유형 정보 조회
    
    Args:
        depression: 우울 여부
        anxiety: 불안 여부
        work_stress: 직무스트레스 여부
        
    Returns:
        스트레스 유형 정보
    """
    try:
        # 임시 파이프라인 (API 키 없이도 동작)
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
        
        stress_type = stress_type_mapping.get(code, StressType.XXX)
        
        # 캐릭터 정보 (API 키 없이도 동작)
        from .models import STRESS_TYPE_PROFILES
        
        stress_profile = STRESS_TYPE_PROFILES.get(stress_type)
        
        return {
            "success": True,
            "stress_type": stress_type.value,
            "stress_code": code,
            "character": {
                "emoji": stress_profile.emoji if stress_profile else "🤖",
                "therapy_methods": [m.korean_name for m in stress_profile.therapy_methods] if stress_profile else []
            },
            "description": _get_stress_type_description(stress_type)
        }
        
    except Exception as e:
        logging.error(f"❌ 스트레스 유형 정보 조회 실패: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_available_occupations() -> List[str]:
    """사용 가능한 직군 목록 반환"""
    from .models import OCCUPATION_KEYWORDS
    return list(OCCUPATION_KEYWORDS.keys())


def get_occupation_keywords(occupation: str) -> List[str]:
    """특정 직군의 키워드 목록 반환"""
    from .models import OCCUPATION_KEYWORDS
    return OCCUPATION_KEYWORDS.get(occupation, [])


def health_check(openai_api_key: Optional[str] = None) -> Dict[str, Any]:
    """시스템 상태 확인"""
    try:
        pipeline = get_pipeline(openai_api_key=openai_api_key)
        return pipeline.health_check()
    except Exception as e:
        logging.error(f"❌ 시스템 상태 확인 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "overall_status": False
        }


def get_system_info() -> Dict[str, Any]:
    """시스템 정보 조회 (API 키 불필요)"""
    try:
        from .models import STRESS_TYPE_PROFILES, OCCUPATION_KEYWORDS
        
        return {
            "success": True,
            "pipeline_version": "1.0.0-production",
            "stress_types": {stress_type.name: stress_type.value for stress_type in StressType},
            "stress_types_count": len(STRESS_TYPE_PROFILES),
            "occupations_count": len(OCCUPATION_KEYWORDS),
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
        logging.error(f"❌ 시스템 정보 조회 실패: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def _get_stress_type_description(stress_type: StressType) -> str:
    """스트레스 유형별 설명 반환"""
    descriptions = {
        StressType.XXX: "평온한 상태로, 마음챙김을 통한 현재 상태 유지가 도움됩니다.",
        StressType.OXX: "우울감을 경험하고 있으며, 긍정심리치료와 마음챙김이 효과적입니다.",
        StressType.XOX: "불안감을 느끼고 있으며, 수용전념치료와 마음챙김이 도움됩니다.",
        StressType.XXO: "직무 관련 스트레스를 경험하고 있으며, 수용전념치료와 인지행동치료가 효과적입니다.",
        StressType.OOX: "우울과 불안을 동시에 경험하고 있으며, 긍정심리치료와 수용전념치료가 필요합니다.",
        StressType.OXO: "우울과 직무스트레스를 겪고 있으며, 긍정심리치료와 수용전념치료가 도움됩니다.",
        StressType.XOO: "불안과 직무스트레스를 경험하고 있으며, 수용전념치료와 인지행동치료가 효과적입니다.",
        StressType.OOO: "복합적인 스트레스 상황으로, 긍정심리치료, 수용전념치료, 변증법적행동치료가 모두 필요합니다."
    }
    
    return descriptions.get(stress_type, "스트레스 상태에 대한 정보를 찾을 수 없습니다.")


# 편의 함수들
def create_user_profile_simple(nickname: str,
                              age: int,
                              gender: str,
                              occupation: str,
                              depression: bool = False,
                              anxiety: bool = False,
                              work_stress: bool = False,
                              personal_keywords: Optional[List[str]] = None) -> UserProfile:
    """간단한 사용자 프로필 생성"""
    if personal_keywords is None:
        personal_keywords = []
    
    # 스트레스 유형 결정
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
    
    stress_type = stress_type_mapping.get(code, StressType.XXX)
    
    return UserProfile(
        nickname=nickname,
        age=age,
        gender=gender,
        occupation=occupation,
        stress_type=stress_type,
        survey_features=["기본_설문_기반"],  # 기본값 설정
        personal_keywords=personal_keywords,
        msi=75.0,
        psi=68.0
    )


def reset_pipeline():
    """글로벌 파이프라인 인스턴스 리셋"""
    global _pipeline_instance
    _pipeline_instance = None