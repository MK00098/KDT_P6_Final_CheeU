#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU VectorDB Module
벡터 데이터베이스 관리 및 검색 기능
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangChain 컴포넌트
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# ChromaDB
import chromadb
from chromadb.config import Settings

# .env 파일 자동 로드
load_dotenv()


class CheeUVectorDB:
    """
    CheeU VectorDB 관리 클래스
    - ChromaDB 기반 벡터 저장소
    - 한국어 임베딩 모델 사용
    - 우선순위 기반 검색
    """
    
    def __init__(self, vector_db_path: str = "../papers/vectordb"):
        """
        VectorDB 초기화
        
        Args:
            vector_db_path: ChromaDB 저장 경로
        """
        self.vector_db_path = vector_db_path
        self.logger = logging.getLogger(__name__)
        
        # 한국어 임베딩 모델 설정 (384차원)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # ChromaDB 설정
        self.client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Chroma VectorStore 초기화
        self.vector_store = None
        self._init_vector_store()
    
    def _init_vector_store(self):
        """벡터 저장소 초기화"""
        try:
            self.vector_store = Chroma(
                client=self.client,
                embedding_function=self.embeddings,
                collection_name="therapy_content"
            )
            self.logger.info(f"✅ VectorDB 로드 완료: {self.vector_db_path}")
        except Exception as e:
            self.logger.error(f"❌ VectorDB 로드 실패: {e}")
            raise
    
    def search_basic(self, query: str, k: int = 3) -> List[Document]:
        """
        기본 유사도 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            관련 문서 리스트
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            self.logger.info(f"🔍 기본 검색 완료: {len(docs)}개 문서")
            return docs
        except Exception as e:
            self.logger.error(f"❌ 검색 오류: {e}")
            return []
    
    def search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """
        점수 포함 유사도 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            (문서, 점수) 튜플 리스트
        """
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            self.logger.info(f"🔍 점수 포함 검색 완료: {len(docs_with_scores)}개 문서")
            return docs_with_scores
        except Exception as e:
            self.logger.error(f"❌ 검색 오류: {e}")
            return []
    
    def search_with_priority_weighting(self, 
                                     main_query: str,
                                     sub_queries: List[str],
                                     main_weight: float = 0.7,
                                     sub_weight: float = 0.3,
                                     k: int = 3) -> List[Document]:
        """
        우선순위 가중치 기반 검색
        
        Args:
            main_query: 메인 검색 쿼리 (사용자 입력 + 스트레스 유형)
            sub_queries: 서브 검색 쿼리들 (연령, 성별, 직군, 키워드)
            main_weight: 메인 쿼리 가중치
            sub_weight: 서브 쿼리 가중치
            k: 반환할 문서 수
            
        Returns:
            우선순위 조정된 관련 문서 리스트
        """
        try:
            # 1. 메인 쿼리로 검색
            main_docs_with_scores = self.vector_store.similarity_search_with_score(
                main_query, k=k*2
            )
            
            # 2. 서브 쿼리들로 검색
            sub_docs_with_scores = []
            for sub_query in sub_queries:
                if sub_query.strip():
                    sub_results = self.vector_store.similarity_search_with_score(
                        sub_query, k=k
                    )
                    sub_docs_with_scores.extend(sub_results)
            
            # 3. 가중치 적용 및 통합
            doc_scores = {}
            
            # 메인 쿼리 점수 적용
            for doc, score in main_docs_with_scores:
                doc_id = doc.page_content[:100]  # 문서 식별용
                doc_scores[doc_id] = {
                    'doc': doc,
                    'score': score * main_weight,
                    'main_score': score
                }
            
            # 서브 쿼리 점수 적용
            for doc, score in sub_docs_with_scores:
                doc_id = doc.page_content[:100]
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += score * (sub_weight / len(sub_queries))
                else:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'score': score * (sub_weight / len(sub_queries)),
                        'main_score': 1.0
                    }
            
            # 4. 점수순 정렬 및 상위 k개 반환 (높은 점수부터)
            sorted_docs = sorted(
                doc_scores.values(), 
                key=lambda x: x['score'],
                reverse=True
            )[:k]
            
            result_docs = [item['doc'] for item in sorted_docs]
            
            self.logger.info(f"🎯 우선순위 검색 완료: {len(result_docs)}개 문서")
            return result_docs
            
        except Exception as e:
            self.logger.error(f"❌ 우선순위 검색 오류: {e}")
            # 폴백: 기본 검색
            return self.search_basic(main_query, k)
    
    def calculate_search_confidence(self, docs: List[Document], 
                                  keywords: List[str]) -> float:
        """
        검색 결과 신뢰도 계산
        
        Args:
            docs: 검색된 문서들
            keywords: 사용자 키워드들
            
        Returns:
            신뢰도 점수 (0.0-1.0)
        """
        if not docs:
            return 0.0
        
        total_score = 0.0
        keyword_matches = 0
        
        for doc in docs:
            content = doc.page_content.lower()
            
            # 키워드 매칭 점수
            for keyword in keywords:
                if keyword.lower() in content:
                    keyword_matches += 1
            
            # 문서 길이 점수 (적절한 길이)
            content_length = len(content)
            if 100 <= content_length <= 1000:
                total_score += 0.3
            elif content_length > 50:
                total_score += 0.1
        
        # 키워드 매칭 비율
        if keywords:
            keyword_ratio = keyword_matches / (len(keywords) * len(docs))
            total_score += keyword_ratio * 0.7
        
        # 문서 수 보정
        doc_count_score = min(len(docs) / 3.0, 1.0) * 0.2
        total_score += doc_count_score
        
        return min(total_score, 1.0)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 조회
        
        Returns:
            컬렉션 메타데이터
        """
        try:
            collection = self.client.get_collection("therapy_content")
            count = collection.count()
            
            return {
                "collection_name": "therapy_content",
                "document_count": count,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "db_path": self.vector_db_path,
                "status": "active"
            }
        except Exception as e:
            self.logger.error(f"❌ 컬렉션 정보 조회 실패: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def health_check(self) -> bool:
        """
        VectorDB 상태 확인
        
        Returns:
            정상 상태 여부
        """
        try:
            # 간단한 테스트 검색
            test_docs = self.search_basic("스트레스", k=1)
            return len(test_docs) > 0
        except Exception as e:
            self.logger.error(f"❌ VectorDB 상태 확인 실패: {e}")
            return False


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    vectordb = CheeUVectorDB()
    
    # 상태 확인
    print("🔍 VectorDB 상태 확인:")
    print(f"정상 상태: {vectordb.health_check()}")
    
    # 컬렉션 정보
    print("\n📊 컬렉션 정보:")
    info = vectordb.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 기본 검색 테스트
    print("\n🔍 기본 검색 테스트:")
    docs = vectordb.search_basic("우울 스트레스", k=2)
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {doc.page_content[:100]}...")
    
    # 우선순위 검색 테스트
    print("\n🎯 우선순위 검색 테스트:")
    priority_docs = vectordb.search_with_priority_weighting(
        main_query="우울 무기력",
        sub_queries=["20대", "여성", "의료진", "피로"],
        k=2
    )
    for i, doc in enumerate(priority_docs):
        print(f"  [{i+1}] {doc.page_content[:100]}...")