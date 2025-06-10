from typing import List, Optional
from .models import Query, Response, SearchResult

class ResponseGenerator:

    def __init__(self):
        pass
        
    def generate_response(self, query: Query, sources: List[SearchResult]) -> Response:

        answer = self._generate_answer(query, sources)
        
        confidence = self._calculate_confidence(query, answer, sources)
        
        return Response(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence
        )
        
    def _generate_answer(self, query: Query, sources: List[SearchResult]) -> str:

        return ""
        
    def _calculate_confidence(self, query: Query, answer: str, sources: List[SearchResult]) -> float:

        return 0.0 