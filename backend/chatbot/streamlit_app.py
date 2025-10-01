#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU RAG+LLM Pipeline - Streamlit Test Interface
간단한 웹 인터페이스로 CheeU 파이프라인 테스트
"""

import streamlit as st
import os
import sys
from pathlib import Path

# CheeU 패키지 임포트
try:
    from cheeu_rag import quick_healing_message, health_check, get_stress_type_info
    from cheeu_rag.models import StressType
except ImportError as e:
    st.error(f"CheeU 패키지 임포트 오류: {e}")
    st.error("requirements.txt의 의존성을 설치해주세요: pip install -r requirements.txt")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="CheeU RAG+LLM Pipeline 테스트",
    page_icon="🧠",
    layout="wide"
)

# 스타일 설정
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E8B57;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.stress-type {
    font-size: 1.2rem;
    color: #FF6B6B;
    font-weight: bold;
}
.healing-message {
    background-color: #F0F8FF;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #2E8B57;
}
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown('<h1 class="main-header">🧠 CheeU RAG+LLM Pipeline 테스트</h1>', unsafe_allow_html=True)
st.markdown("---")

# 사이드바 - 시스템 상태 체크
with st.sidebar:
    st.header("🔧 시스템 상태")
    
    if st.button("시스템 상태 확인"):
        try:
            with st.spinner("시스템 상태 확인 중..."):
                health_result = health_check()
            
            if health_result["status"] == "healthy":
                st.success("✅ 시스템 정상")
                st.json(health_result)
            else:
                st.error("❌ 시스템 오류")
                st.json(health_result)
        except Exception as e:
            st.error(f"상태 확인 실패: {e}")

    st.markdown("---")
    st.markdown("### 📋 스트레스 유형")
    for stress_type in StressType:
        emoji_map = {
            "평온형": "🦥",
            "우울형": "🐻", 
            "불안형": "🐰",
            "직무스트레스형": "🦔",
            "우울+불안형": "🦌",
            "우울+직무스트레스형": "🦫",
            "불안+직무스트레스형": "🐿️",
            "위기형": "🦊"
        }
        emoji = emoji_map.get(stress_type.value, "🎭")
        st.write(f"{emoji} {stress_type.value}")

# 메인 인터페이스
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 치유 메시지 생성")
    
    # API 키 입력
    api_key = st.text_input("OpenAI API 키", type="password", help="sk-으로 시작하는 OpenAI API 키를 입력하세요")
    
    if not api_key:
        st.warning("⚠️ OpenAI API 키를 입력해주세요")
        st.stop()

    # 사용자 입력
    user_input = st.text_area(
        "고민이나 스트레스를 자유롭게 적어주세요",
        placeholder="예: 요즘 너무 스트레스 받아서 힘들어요...",
        height=100
    )
    
    # 사용자 정보
    st.subheader("👤 사용자 정보")
    col_name, col_age = st.columns(2)
    with col_name:
        nickname = st.text_input("닉네임", value="사용자")
    with col_age:
        age = st.number_input("나이", min_value=10, max_value=100, value=25)
    
    col_gender, col_occupation = st.columns(2)
    with col_gender:
        gender = st.selectbox("성별", ["남성", "여성", "기타"])
    with col_occupation:
        occupation = st.selectbox("직업", [
            "보건·의료", "교육·자연과학·사회과학", "경영·회계·사무", 
            "정보통신", "건설", "사회복지·종교", "문화·예술·디자인·방송",
            "금융·보험", "기타"
        ])
    
    # 스트레스 유형
    st.subheader("😰 스트레스 유형")
    col_dep, col_anx, col_work = st.columns(3)
    with col_dep:
        depression = st.checkbox("우울")
    with col_anx:
        anxiety = st.checkbox("불안")
    with col_work:
        work_stress = st.checkbox("직무스트레스")
    
    # 설문 키워드 (설문 응답 기반)
    survey_keywords = st.multiselect(
        "설문 키워드 (설문 응답에서 추출된 스트레스 관련 증상)",
        ["피로감", "번아웃", "압박감", "무기력", "집중력 저하", "수면 문제", "대인관계", "업무과부하"]
    )

with col2:
    st.header("📊 결과")
    
    # 스트레스 유형 정보
    if depression or anxiety or work_stress:
        try:
            stress_info = get_stress_type_info(depression, anxiety, work_stress)
            st.markdown(f'<div class="stress-type">{stress_info["stress_type"]} {stress_info["character"]}</div>', 
                       unsafe_allow_html=True)
            st.write(f"**치료법**: {', '.join(stress_info['character']['therapy_methods'])}")
        except Exception as e:
            st.error(f"스트레스 유형 분석 오류: {e}")

# 치유 메시지 생성
if st.button("🔮 치유 메시지 생성", type="primary"):
    if not user_input.strip():
        st.warning("고민이나 스트레스 내용을 입력해주세요")
    else:
        try:
            with st.spinner("AI가 개인화된 치유 메시지를 생성하고 있습니다..."):
                result = quick_healing_message(
                    user_input=user_input,
                    nickname=nickname,
                    age=age,
                    gender=gender,
                    occupation=occupation,
                    depression=depression,
                    anxiety=anxiety,
                    work_stress=work_stress,
                    survey_keywords=survey_keywords,
                    openai_api_key=api_key
                )
            
            if result["success"]:
                healing_data = result["result"]
                
                # 치유 메시지 표시
                st.markdown('<div class="healing-message">', unsafe_allow_html=True)
                st.markdown(f"### {healing_data['character']}")
                st.markdown(f"**{healing_data['healing_message']}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 상세 정보
                with st.expander("상세 정보 보기"):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.write(f"**스트레스 유형**: {healing_data['stress_type']}")
                        st.write(f"**사용된 치료법**: {', '.join(healing_data['therapy_methods_used'])}")
                    with col_info2:
                        st.write(f"**신뢰도**: {healing_data['confidence_score']:.2f}")
                        st.write(f"**생성 시간**: {healing_data['timestamp']}")
                    
                    if healing_data.get('sources'):
                        st.write("**참조 논문**:")
                        for i, source in enumerate(healing_data['sources'][:3], 1):
                            st.write(f"{i}. {source}")
                    
                    if healing_data.get('keywords_used'):
                        st.write(f"**사용된 키워드**: {', '.join(healing_data['keywords_used'])}")
                
            else:
                st.error(f"치유 메시지 생성 실패: {result.get('error', '알 수 없는 오류')}")
                
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            st.error("API 키가 올바른지, 네트워크 연결이 정상인지 확인해주세요")

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🧠 CheeU RAG+LLM Pipeline v1.0.0</p>
    <p>개인화된 멘탈헬스 AI 치유캡슐 생성 시스템</p>
</div>
""", unsafe_allow_html=True)
