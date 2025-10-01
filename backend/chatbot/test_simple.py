#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU Pipeline 간단한 테스트
"""

import os
from cheeu_rag import quick_healing_message

def test_lee_persona():
    """이대리 페르소나 테스트"""

    print("🧪 이대리 페르소나 테스트 시작\n")

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정해주세요.")
        return

    # 테스트 케이스
    test_case = {
        "user_input": "프로젝트 데드라인에 쫓겨서 밤새 작업하는 날이 많아졌어요. 불안하고 집중이 안 되고 실수도 늘어나서 스트레스가 심해요.",
        "nickname": "이대리",
        "age": 27,
        "gender": "여성",
        "occupation": "19. 정보통신",
        "depression": False,
        "anxiety": True,
        "work_stress": True,
        "survey_keywords": ["피로감", "번아웃", "압박감", "집중력 저하", "수면 문제"],
        "openai_api_key": api_key
    }

    print("📋 테스트 프로필:")
    print(f"  • 이름: {test_case['nickname']}")
    print(f"  • 나이: {test_case['age']}세")
    print(f"  • 성별: {test_case['gender']}")
    print(f"  • 직업: {test_case['occupation']}")
    print(f"  • 스트레스: 불안({test_case['anxiety']}), 직무({test_case['work_stress']})")
    print(f"  • 설문키워드: {', '.join(test_case['survey_keywords'])}")
    print(f"\n💬 고민 내용:\n  {test_case['user_input']}\n")

    print("⏳ 치유 메시지 생성 중...")

    try:
        # API 호출
        result = quick_healing_message(**test_case)

        if result["success"]:
            healing_data = result["result"]

            print("\n✅ 치유 메시지 생성 성공!\n")
            print("="*50)
            print(f"🎭 캐릭터: {healing_data['character']}")
            print(f"📝 스트레스 유형: {healing_data['stress_type']}")
            print(f"💊 치료법: {', '.join(healing_data['therapy_methods_used'])}")
            print(f"📊 신뢰도: {healing_data['confidence_score']:.2f}")
            print("="*50)
            print(f"\n💬 치유 메시지:\n")
            print(healing_data['healing_message'])
            print("="*50)

            if healing_data.get('sources'):
                print(f"\n📚 참조 논문:")
                for i, source in enumerate(healing_data['sources'][:3], 1):
                    print(f"  {i}. {source}")

            if healing_data.get('keywords_used'):
                print(f"\n🏷️ 활용 키워드: {', '.join(healing_data['keywords_used'])}")

        else:
            print(f"\n❌ 생성 실패: {result.get('error', '알 수 없는 오류')}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

    print("\n✨ 테스트 완료!")

if __name__ == "__main__":
    test_lee_persona()