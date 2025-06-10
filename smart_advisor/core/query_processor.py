from typing import Dict, List, Optional
import numpy as np
from ..models import Query, Response, SearchResult
from ..services.embedding import EmbeddingService
from ..core.database import VectorDatabase
from ..models import BertModel

class QueryProcessor:
    def __init__(self, embedding_service: EmbeddingService, vector_db: VectorDatabase):
        self._cache = {}
        self._embedding_service = embedding_service
        self._vector_db = vector_db
        self._bert_model = BertModel()
    
    def process_query(self, query: Query) -> Response:
        """
        
        Args:
            query (Query): Входящий запрос для обработки
            
        Returns:
            Response: Обработанный ответ
        """
        if query.text in self._cache:
            return self._cache[query.text]
            
        if not self._validate_query(query):
            return Response(
                answer="Извините, я не могу обработать этот запрос. Пожалуйста, переформулируйте его.",
                sources=[],
                confidence=0.0
            )
            
        query_embedding = self._bert_model.get_embedding(query.text)
        
        search_results = self._vector_db.search(query_embedding, k=3)
        
        response = Response(
            answer=self._generate_answer(query, search_results),
            sources=[result.metadata for result in search_results],
            confidence=self._calculate_confidence(search_results)
        )
        
        self._cache[query.text] = response
        return response
    
    def _preprocess_query(self, query: Query) -> Dict:
        """
        Предварительная обработка запроса
        
        Args:
            query (Query): Исходный запрос
            
        Returns:
            Dict: Обработанные данные запроса
        """
        return {
            'text': query.text.lower().strip(),
            'type': query.type if hasattr(query, 'type') else None
        }
    
    def _validate_query(self, query: Query) -> bool:
        """
        Проверяет валидность запроса
        
        Args:
            query (Query): Запрос для проверки
            
        Returns:
            bool: True если запрос валиден, False в противном случае
        """
        if not query.text or len(query.text.strip()) == 0:
            return False
        return True
    
    def _generate_answer(self, query: Query, search_results: List[SearchResult]) -> str:
        """
        Генерирует ответ на основе найденных документов
        
        Args:
            query (Query): Исходный запрос
            search_results (List[SearchResult]): Результаты поиска
            
        Returns:
            str: Сгенерированный ответ
        """
        if not search_results:
            return "Извините, я не нашел информации по вашему запросу."
            
        return search_results[0].text
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """
        Уверенность в ответе
        
        Args:
            search_results (List[SearchResult]): Результаты поиска
            
        Returns:
            float: Значение уверенности от 0 до 1
        """
        if not search_results:
            return 0.0
            
        return float(max(result.score for result in search_results)) 