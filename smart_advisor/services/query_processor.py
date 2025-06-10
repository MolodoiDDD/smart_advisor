import re
import logging
from typing import List, Optional, Dict, Any, Union
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from ..models import SearchResult, Query, Response
from ..config.config import config

logger = logging.getLogger(__name__)

class QueryProcessor:

    def __init__(self, vector_db):
        self.vector_db = vector_db
        self._cache: Dict[str, Response] = {}
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words("russian"))
        
        self.number_patterns = [
            (r'(\d+)\s*руб(?:лей)?(?:\s+в\s+месяц)?', 'monthly'),
            (r'(\d+)\s*₽(?:\s+в\s+месяц)?', 'monthly'),
            (r'(\d+)\s*р\.(?:\s+в\s+месяц)?', 'monthly'),
            (r'(\d+)\s*руб\.(?:\s+в\s+месяц)?', 'monthly'),
            (r'(\d+)\s*руб(?:лей)?(?:\s+в\s+семестр)?', 'semester'),
            (r'(\d+)\s*₽(?:\s+в\s+семестр)?', 'semester'),
        ]
        
        self.advisor_types = {
            'академическая': [
                'обычная', 'базовая', 'академическая', 'стандартная',
                'основная', 'регулярная', 'ежемесячная'
            ],
            'повышенная': [
                'повышенная', 'увеличенная', 'высокая', 'премиальная',
                'надбавка', 'дополнительная', 'стимулирующая'
            ],
            'социальная': [
                'социальная', 'соц', 'для нуждающихся', 'материальная помощь',
                'поддержка', 'льготная', 'для малоимущих'
            ],
            'специальная': [
                'специальная', 'особая', 'для отличников', 'именная',
                'целевая', 'персональная', 'для талантливых'
            ]
        }
        
        self.question_types = {
            'definition': [
                'что такое', 'определение', 'понятие', 'означает',
                'расскажи про', 'объясни', 'описание', 'характеристика'
            ],
            'amount': [
                'сколько', 'размер', 'сумма', 'выплата', 'начисление',
                'получать', 'выплачивается', 'начисляется', 'стоимость'
            ],
            'requirements': [
                'требования', 'условия', 'критерии', 'кто может получить',
                'нужно', 'необходимо', 'документы', 'список', 'перечень'
            ],
            'deadline': [
                'срок', 'когда', 'период', 'дата', 'время',
                'до какого числа', 'когда подавать', 'когда получать'
            ],
            'procedure': [
                'как получить', 'процедура', 'порядок', 'оформление',
                'шаги', 'этапы', 'процесс', 'алгоритм', 'инструкция'
            ]
        }

    def process_query(self, query_text: str) -> Union[Response, str]:
        try:
            query = Query(text=query_text)
            
            if query.text in self._cache:
                return self._cache[query.text]
                
            query.text = self._preprocess_text(query.text)
            
            if not query.text:
                return Response(
                    query=query,
                    answer="Пожалуйста, введите ваш вопрос.",
                    sources=[],
                    confidence=0.0
                )
                
            search_results = self.vector_db.search(query.text, k=10)
            
            if not search_results:
                return Response(
                    query=query,
                    answer="К сожалению, я не нашел информации по вашему запросу.",
                    sources=[],
                    confidence=0.0
                )
            
            answer, confidence = self._extract_answer_with_context(query.text, search_results)
            
            response = Response(
                query=query,
                answer=answer if answer else "К сожалению, я не смог найти точный ответ на ваш вопрос в доступных документах.",
                sources=search_results[:3],
                confidence=confidence
            )
            
            self._cache[query.text] = response
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {str(e)}")
            return Response(
                query=Query(text=query_text),
                answer="Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте переформулировать вопрос.",
                sources=[],
                confidence=0.0
            )

    def _preprocess_text(self, text: str) -> str:

        text = text.lower().strip()
        
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def _extract_answer_with_context(self, query: str, results: List[SearchResult]) -> tuple[Optional[str], float]:
        try:
            logger.info(f"Обработка запроса: '{query}'")
            
            if not results:
                return None, 0.0
                
            question_type = self._detect_question_type(query)
            advisor_type = self._detect_advisor_type(query)
            
            combined_text = self._combine_results_with_weights(results)
            
            answer = None
            confidence = 0.0
            
            if question_type == 'definition':
                answer = self._extract_definition(combined_text)
                confidence = 0.8 if answer else 0.3
            elif question_type == 'amount':
                amount, period = self._extract_advisor_amount(combined_text, advisor_type)
                if amount:
                    answer = f"Размер {advisor_type} стипендии составляет {amount} рублей"
                    if period:
                        answer += f" ({period})"
                    confidence = 0.9
            elif question_type == 'requirements':
                requirements = self._extract_requirements(combined_text, advisor_type)
                if requirements:
                    answer = f"Требования для получения {advisor_type} стипендии:\n{requirements}"
                    confidence = 0.85
            elif question_type == 'deadline':
                deadline_info = self._extract_deadline_info(combined_text, advisor_type)
                if deadline_info:
                    answer = deadline_info
                    confidence = 0.8
            elif question_type == 'procedure':
                procedure_info = self._extract_procedure_info(combined_text, advisor_type)
                if procedure_info:
                    answer = procedure_info
                    confidence = 0.85
            
            if not answer:
                answer = self._extract_most_relevant_text(query, results)
                confidence = 0.6
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении ответа: {str(e)}")
            return None, 0.0

    def _combine_results_with_weights(self, results: List[SearchResult]) -> str:
        combined = []
        for result in results:
            combined.append(result.text)
        return "\n".join(combined)

    def _extract_most_relevant_text(self, query: str, results: List[SearchResult]) -> str:
        query_tokens = set(self.stemmer.stem(word) for word in query.split() 
                         if word not in self.stop_words)
        
        best_sentences = []
        max_score = 0
        
        for result in results:
            sentences = sent_tokenize(result.text, language='russian')
            
            for sentence in sentences:
                sentence_tokens = set(self.stemmer.stem(word) for word in sentence.split() 
                                   if word not in self.stop_words)
                
                matching_tokens = query_tokens & sentence_tokens
                score = len(matching_tokens) / len(query_tokens) if query_tokens else 0
                
                if score > max_score:
                    max_score = score
                    best_sentences = [sentence]
                elif score == max_score:
                    best_sentences.append(sentence)
        
        if best_sentences:
            return " ".join(best_sentences[:3])
        
        return results[0].text[:500] + "..."

    def _extract_advisor_amount(self, text: str, advisor_type: str) -> tuple[Optional[str], Optional[str]]:
        amounts = []
        periods = []
        
        for pattern, period_type in self.number_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                amount = match.group(1)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                if any(keyword in context for keyword in self.advisor_types[advisor_type]):
                    amounts.append(amount)
                    periods.append(period_type)
        
        if amounts:
            max_amount = max(amounts, key=int)
            period = periods[amounts.index(max_amount)]
            return max_amount, period
            
        return None, None

    def _extract_definition(self, text: str) -> Optional[str]:
        sentences = sent_tokenize(text, language='russian')
        definition_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in self.question_types['definition']):
                definition_sentences.append(sentence)
        
        if definition_sentences:
            return " ".join(definition_sentences)
            
        return None

    def _detect_question_type(self, query: str) -> str:
        query = query.lower()
        
        type_scores = Counter()
        
        for q_type, keywords in self.question_types.items():
            for keyword in keywords:
                if keyword in query:
                    type_scores[q_type] += 1
        
        if type_scores:
            return type_scores.most_common(1)[0][0]
            
        logger.info("Тип вопроса не определен, возвращаем 'general'")
        return 'general'

    def _detect_advisor_type(self, query: str) -> str:
        query = query.lower()
        
        type_scores = Counter()
        
        for s_type, keywords in self.advisor_types.items():
            for keyword in keywords:
                if keyword in query:
                    type_scores[s_type] += 1
        
        if type_scores:
            return type_scores.most_common(1)[0][0]
            
        logger.info("Тип стипендии не определен, возвращаем 'regular'")
        return 'regular'

    def extract_answer(self, query: str, results: List[SearchResult]) -> Optional[str]:
        try:
            logger.info(f"Обработка запроса: '{query}'")
            
            if not results:
                return None
                
            question_type = self._detect_question_type(query)
            advisor_type = self._detect_advisor_type(query)
            
            combined_text = "\n".join([result.text for result in results])
            
            if question_type == 'definition':
                return self._extract_definition(combined_text)
            elif question_type == 'amount':
                amount = self._extract_advisor_amount(combined_text, advisor_type)
                if amount:
                    return f"Размер {advisor_type} стипендии составляет {amount} рублей."
            elif question_type == 'requirements':
                requirements = self._extract_requirements(combined_text, advisor_type)
                if requirements:
                    return f"Требования для получения {advisor_type} стипендии:\n{requirements}"
            elif question_type == 'deadline':
                deadline_info = self._extract_deadline_info(combined_text, advisor_type)
                if deadline_info:
                    return deadline_info
            elif question_type == 'procedure':
                procedure_info = self._extract_procedure_info(combined_text, advisor_type)
                if procedure_info:
                    return procedure_info
            
            return self._extract_most_relevant_text(query, results[0].text)
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении ответа: {str(e)}")
            return None

    def _extract_deadline_info(self, text: str, advisor_type: str) -> Optional[str]:
        sentences = re.split(r'[.!?]+', text)
        deadline_sentences = []
        
        keywords = ['срок', 'дата', 'период', 'когда', 'выплачивается', 'назначается']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords):
                deadline_sentences.append(sentence)
                
        if deadline_sentences:
            return "\n".join(deadline_sentences)
        return None

    def _extract_procedure_info(self, text: str, advisor_type: str) -> Optional[str]:
        sentences = re.split(r'[.!?]+', text)
        procedure_sentences = []
        
        keywords = ['получить', 'оформить', 'подать', 'заявление', 'документы', 'процедура', 'порядок']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords):
                procedure_sentences.append(sentence)
                
        if procedure_sentences:
            return "\n".join(procedure_sentences)
        return None

    def _extract_requirements(self, text: str, advisor_type: str) -> Optional[str]:
        logger.info(f"Поиск требований для получения {advisor_type} стипендии")
        sentences = re.split(r'[.!?]+', text)
        requirement_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['требование', 'условие', 'критерий']):
                requirement_sentences.append(sentence.strip())
                logger.info(f"Найдено предложение с требованием: {sentence.strip()[:100]}...")
                
        if requirement_sentences:
            logger.info(f"Возвращаем {len(requirement_sentences)} предложений с требованиями")
            return "\n".join(requirement_sentences)
            
        logger.warning("Требования для получения стипендии не найдены в тексте")
        return None 