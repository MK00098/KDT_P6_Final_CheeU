#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheeU Vector DB Module - Production Optimized
벡터 데이터베이스 관리 및 우선순위 기반 검색 기능
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

# LangChain 컴포넌트
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# ChromaDB
import chromadb
from chromadb.config import Settings


class CheeUVectorDB:
    """
    CheeU VectorDB 관리 클래스 - 프로덕션 최적화
    - ChromaDB 기반 벡터 저장소
    - 한국어 임베딩 모델 사용
    - 우선순위 기반 검색 (70/30 가중치)
    - 배치 처리 및 캐싱 지원
    """
    
    def __init__(self,
                 vector_db_path: str = None,
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 collection_name: str = "therapy_content",
                 device: str = "cpu"):
        """
        VectorDB 초기화
        
        Args:
            vector_db_path: ChromaDB 저장 경로 (None시 기본 경로 사용)
            embedding_model: 임베딩 모델명
            collection_name: 컬렉션 이름
            device: 디바이스 ('cpu' 또는 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        
        # 기본 경로 설정
        if vector_db_path is None:
            base_dir = Path(__file__).parent.parent
            vector_db_path = str(base_dir / "data" / "vectordb")
        
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        
        # 디렉토리 생성
        Path(vector_db_path).mkdir(parents=True, exist_ok=True)
        
        # 한국어 임베딩 모델 설정
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # ChromaDB 설정 (프로덕션 최적화)
        self.client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False,  # 프로덕션에서는 리셋 비활성화
                is_persistent=True
            )
        )
        
        # 성능 캐시 초기화
        self._search_cache = {}
        self._cache_max_size = 100
        
        # Chroma VectorStore 초기화
        self.vector_store = None
        self._init_vector_store()
    
    def _init_vector_store(self):
        """벡터 저장소 초기화 (에러 핸들링 강화)"""
        try:
            self.vector_store = Chroma(
                client=self.client,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # 컬렉션 정보 로깅
            collection_info = self.get_collection_info()
            doc_count = collection_info.get("document_count", 0)
            
            if doc_count > 0:
                self.logger.info(f"✅ VectorDB 로드 완료: {self.vector_db_path} ({doc_count}개 문서)")
            else:
                self.logger.warning(f"⚠️ VectorDB 빈 상태: {self.vector_db_path}")
                
        except Exception as e:
            self.logger.error(f"❌ VectorDB 초기화 실패: {e}")
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
            # 캐시 확인
            cache_key = f"basic_{query}_{k}"
            if cache_key in self._search_cache:
                self.logger.debug(f"🎯 캐시 히트: {query[:30]}...")
                return self._search_cache[cache_key]
            
            docs = self.vector_store.similarity_search(query, k=k)
            
            # 캐시 저장 (크기 제한)
            if len(self._search_cache) < self._cache_max_size:
                self._search_cache[cache_key] = docs
            
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
        우선순위 가중치 기반 검색 (핵심 알고리즘)
        
        Args:
            main_query: 메인 검색 쿼리 (사용자 입력 + 스트레스 유형)
            sub_queries: 서브 검색 쿼리들 (연령, 성별, 직군, 키워드)
            main_weight: 메인 쿼리 가중치 (기본값: 0.7)
            sub_weight: 서브 쿼리 가중치 (기본값: 0.3)
            k: 반환할 문서 수
            
        Returns:
            우선순위 조정된 관련 문서 리스트
        """
        try:
            # 캐시 확인
            cache_key = f"priority_{main_query}_{hash(str(sub_queries))}_{k}"
            if cache_key in self._search_cache:
                self.logger.debug(f"🎯 우선순위 캐시 히트: {main_query[:30]}...")
                return self._search_cache[cache_key]
            
            # 1. 메인 쿼리로 검색 (더 많은 후보 확보)
            main_docs_with_scores = self.vector_store.similarity_search_with_score(
                main_query, k=k*3  # 3배 많은 후보
            )
            
            # 2. 서브 쿼리들로 검색
            sub_docs_with_scores = []
            for sub_query in sub_queries:
                if sub_query and sub_query.strip():
                    try:
                        sub_results = self.vector_store.similarity_search_with_score(
                            sub_query, k=k*2  # 2배 후보
                        )
                        sub_docs_with_scores.extend(sub_results)
                    except Exception as e:
                        self.logger.warning(f"서브 쿼리 검색 실패 '{sub_query}': {e}")
                        continue
            
            # 3. 가중치 적용 및 통합
            doc_scores = {}
            
            # 메인 쿼리 점수 적용 (거리 -> 유사도 변환: 1 - distance)
            for doc, distance in main_docs_with_scores:
                doc_id = doc.page_content[:100]  # 문서 식별용
                similarity = max(0, 1 - distance)  # 거리를 유사도로 변환
                doc_scores[doc_id] = {
                    'doc': doc,
                    'score': similarity * main_weight,
                    'main_score': similarity,
                    'sub_score': 0.0
                }
            
            # 서브 쿼리 점수 적용
            sub_weight_per_query = sub_weight / max(len([q for q in sub_queries if q.strip()]), 1)
            for doc, distance in sub_docs_with_scores:
                doc_id = doc.page_content[:100]
                similarity = max(0, 1 - distance)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += similarity * sub_weight_per_query
                    doc_scores[doc_id]['sub_score'] += similarity * sub_weight_per_query
                else:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'score': similarity * sub_weight_per_query,
                        'main_score': 0.0,
                        'sub_score': similarity * sub_weight_per_query
                    }
            
            # 4. 점수순 정렬 및 상위 k개 반환
            sorted_docs = sorted(
                doc_scores.values(), 
                key=lambda x: x['score'],
                reverse=True
            )[:k]
            
            result_docs = [item['doc'] for item in sorted_docs]
            
            # 캐시 저장
            if len(self._search_cache) < self._cache_max_size:
                self._search_cache[cache_key] = result_docs
            
            # 성능 로깅
            avg_score = sum(item['score'] for item in sorted_docs) / len(sorted_docs) if sorted_docs else 0
            self.logger.info(f"🎯 우선순위 검색 완료: {len(result_docs)}개 문서 (평균점수: {avg_score:.3f})")
            
            return result_docs
            
        except Exception as e:
            self.logger.error(f"❌ 우선순위 검색 오류: {e}")
            # 폴백: 기본 검색
            return self.search_basic(main_query, k)
    
    def calculate_search_confidence(self, docs: List[Document], 
                                  keywords: List[str]) -> float:
        """
        검색 결과 신뢰도 계산 (개선된 알고리즘)
        
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
        total_keywords = len(keywords) if keywords else 1
        
        for doc in docs:
            content = doc.page_content.lower()
            doc_score = 0.0
            
            # 1. 키워드 매칭 점수 (가중치: 50%)
            doc_keyword_matches = 0
            for keyword in keywords:
                if keyword and keyword.lower() in content:
                    doc_keyword_matches += 1
            
            keyword_ratio = doc_keyword_matches / total_keywords
            doc_score += keyword_ratio * 0.5
            keyword_matches += doc_keyword_matches
            
            # 2. 문서 길이 점수 (가중치: 20%)
            content_length = len(content)
            if 100 <= content_length <= 1000:
                doc_score += 0.2  # 적절한 길이
            elif 50 <= content_length <= 1500:
                doc_score += 0.1  # 허용 가능한 길이
            
            # 3. 메타데이터 품질 점수 (가중치: 15%)
            metadata = doc.metadata
            if metadata.get('filename'):
                doc_score += 0.1
            if metadata.get('source') == 'research_paper':
                doc_score += 0.05
            
            # 4. 컨텍스트 풍부도 점수 (가중치: 15%)
            sentences = content.count('.')
            if sentences >= 3:
                doc_score += 0.15
            elif sentences >= 1:
                doc_score += 0.08
            
            total_score += doc_score
        
        # 정규화 및 문서 수 보정
        avg_score = total_score / len(docs)
        doc_count_bonus = min(len(docs) / 3.0, 1.0) * 0.1
        
        final_confidence = min(avg_score + doc_count_bonus, 1.0)
        
        self.logger.debug(f"📊 신뢰도 계산: {final_confidence:.3f} ({len(docs)}개 문서, {keyword_matches}/{total_keywords*len(docs)} 키워드)")
        
        return final_confidence
    
    def add_documents(self, documents: List[Document], batch_size: int = 50):
        """
        문서 배치 추가 (프로덕션 최적화)
        
        Args:
            documents: 추가할 문서들
            batch_size: 배치 크기
        """
        try:
            # 배치 단위로 처리
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                self.vector_store.add_documents(batch)
                self.logger.info(f"📥 문서 배치 추가: {i+1}-{min(i+batch_size, len(documents))}/{len(documents)}")
            
            # 캐시 클리어
            self._search_cache.clear()
            self.logger.info(f"✅ 총 {len(documents)}개 문서 추가 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 문서 추가 실패: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 조회 (상세 정보 포함)
        
        Returns:
            컬렉션 메타데이터
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            # 추가 통계 계산
            if count > 0:
                # 샘플 문서 조회
                sample_docs = self.search_basic("테스트", k=1)
                avg_doc_length = len(sample_docs[0].page_content) if sample_docs else 0
            else:
                avg_doc_length = 0
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "db_path": self.vector_db_path,
                "avg_document_length": avg_doc_length,
                "cache_size": len(self._search_cache),
                "status": "active",
                "created_at": "2024-01-01T00:00:00"
            }
            
        except Exception as e:
            self.logger.error(f"❌ 컬렉션 정보 조회 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "document_count": 0
            }
    
    def health_check(self) -> bool:
        """
        VectorDB 상태 확인 (강화된 헬스체크)
        
        Returns:
            정상 상태 여부
        """
        try:
            # 1. 컬렉션 접근 확인
            info = self.get_collection_info()
            if info.get("status") == "error":
                return False
            
            # 2. 검색 기능 확인
            test_docs = self.search_basic("테스트", k=1)
            
            # 3. 문서 수 확인
            doc_count = info.get("document_count", 0)
            if doc_count == 0:
                self.logger.warning("⚠️ VectorDB에 문서가 없습니다")
                return False
            
            self.logger.info(f"✅ VectorDB 정상 상태: {doc_count}개 문서")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ VectorDB 상태 확인 실패: {e}")
            return False
    
    def clear_cache(self):
        """검색 캐시 클리어"""
        self._search_cache.clear()
        self.logger.info("🧹 검색 캐시 클리어 완료")