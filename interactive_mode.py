import logging
from smart_advisor.services.file_loader import FileLoader
from smart_advisor.core.embeddings import EmbeddingService
from smart_advisor.services.gpt_service import GPTService
from smart_advisor.core.database import VectorDatabase
from smart_advisor.config import settings
from smart_advisor.core.models import Document
import re
import string
import os
import sys
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

print("Запуск приложения...")
print("Инициализация компонентов...")

RUSSIAN_STOP_WORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
    'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
    'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
    'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'чего',
    'при', 'на', 'об', 'за', 'что', 'то', 'этот', 'тот', 'такой', 'такая', 'такое', 'такие',
    'эти', 'эта', 'это', 'этот', 'который', 'которого', 'которому', 'которым', 'которую',
    'которые', 'которых', 'которыми', 'кто', 'кого', 'кому', 'кем', 'где', 'когда', 'как',
    'какой', 'какая', 'какое', 'какие', 'сколько', 'чей', 'чья', 'чье', 'чьи', 'почему',
    'зачем', 'как', 'какой', 'какая', 'какое', 'какие', 'сколько', 'чей', 'чья', 'чье', 'чьи',
    'почему', 'зачем'
}

SYNONYMS = {
    'стипендия': ['стипендия', 'стипендиальный', 'стипендиат', 'стипендию', 'стипендией', 'стипендию'],
    'социальная': ['социальная', 'социальный', 'социально', 'социальную', 'социальным', 'социальной'],
    'академическая': ['академическая', 'академический', 'академически', 'академическую', 'академическим', 'академической'],
    'президентская': ['президентская', 'президентский', 'президентски', 'президентскую', 'президентским', 'президентской'],
    'размер': ['размер', 'сумма', 'величина', 'объем', 'количество', 'сколько'],
    'рубль': ['рубль', 'руб', 'рублей', '₽', 'руб.', 'рублях', 'рублями'],
    'документ': ['документ', 'справка', 'заявление', 'заявка', 'обращение', 'документы', 'документами'],
    'деканат': ['деканат', 'декана', 'деканату', 'деканатом', 'деканате', 'деканата', 'деканаты'],
    'срок': ['срок', 'дата', 'период', 'время', 'когда', 'до', 'после', 'в течение'],
    'подача': ['подача', 'предоставление', 'передача', 'отправка', 'доставка', 'подать', 'подавать', 'подал'],
    'задолженность': ['задолженность', 'долг', 'долги', 'задолженности', 'задолженностями', 'задолженностям'],
    'сдать': ['сдать', 'сдавать', 'сдал', 'сдаю', 'сдаем', 'сдаете', 'сдают', 'пересдать', 'пересдавать'],
    'академический': ['академический', 'академическая', 'академическое', 'академические', 'академическим', 'академической'],
    'университет': ['университет', 'вуз', 'институт', 'университета', 'университету', 'университетом', 'университете'],
    'студент': ['студент', 'студентка', 'студенты', 'студентки', 'студента', 'студенке', 'студентом', 'студенткой']
}

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

def extract_keywords(query: str) -> list:
    words = query.lower().split()

    keywords = [word for word in words if word not in RUSSIAN_STOP_WORDS and word not in string.punctuation]

    if not keywords:
        keywords = [word for word in words if word not in string.punctuation]

    expanded_keywords = []
    for keyword in keywords:
        expanded_keywords.append(keyword)
        for base_word, synonyms in SYNONYMS.items():
            if keyword in synonyms:
                expanded_keywords.extend(synonyms)
    
    return expanded_keywords

def extract_key_information(text: str, query: str) -> str:

    keywords = extract_keywords(query)

    sentences = re.split(r'[.!?]+', text)

    sentence_scores = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_words = sentence.lower().split()

        keyword_count = sum(1 for word in sentence_words if word in keywords)

        normalized_score = keyword_count / len(sentence_words) if sentence_words else 0

        if any(word in sentence.lower() for word in query.lower().split()):
            normalized_score *= 1.5
        
        sentence_scores.append((sentence, normalized_score))

    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    

    relevant_sentences = [s[0] for s in sentence_scores[:3] if s[1] > 0.1]
    

    if relevant_sentences:
        return ' '.join(relevant_sentences)
    

    return sentences[0] if sentences else text

def format_response(query: str, results: list) -> str:
    if not results:
        return "К сожалению, я не нашел информации по вашему запросу. Попробуйте переформулировать вопрос."

    relevant_info = []
    for res in results:
        similarity = 1 - res.score
        if similarity > 0.5:
            text = res.text
            source = res.document.metadata.get('source', 'неизвестный источник')
            relevant_info.append({
                'text': text,
                'source': source,
                'similarity': similarity
            })

    if not relevant_info:
        return "Найдена информация, но она недостаточно релевантна вашему запросу. Попробуйте уточнить вопрос."

    relevant_info.sort(key=lambda x: x['similarity'], reverse=True)
    top_results = relevant_info[:3]

    response = []
    for info in top_results:
        key_info = extract_key_information(info['text'], query)
        if key_info:
            response.append(f"{key_info} (источник: {info['source']})")

    if not response:
        return "Не удалось извлечь релевантную информацию из найденных документов. Попробуйте переформулировать вопрос."

    return "\n\n".join(response)

def initialize_components():
    try:
        print("\n=== Инициализация компонентов ===")
        logger.info("Инициализация компонентов...")

        for dir_path in [settings.DATA_DIR, settings.HTML_DIR, settings.PDF_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"Создана директория: {dir_path}")
                logger.info(f"Создана директория: {dir_path}")

        print("\nСоздание сервисов...")
        file_loader = FileLoader()
        print("✓ FileLoader создан")
        
        embedder = EmbeddingService()
        print("✓ EmbeddingService создан")
        
        vector_db = VectorDatabase(embedder)
        print("✓ VectorDatabase создана")
        
        gpt_service = GPTService()
        print("✓ GPTService создан")
        
        print("\nЗагрузка документов...")
        documents = file_loader.load_documents()
        
        if not documents:
            print("⚠️ Документы не найдены")
            logger.warning("Документы не найдены")
            return None, None, None, None
        
        print(f"✓ Загружено {len(documents)} документов")
        logger.info(f"Загружено {len(documents)} документов")
        
        print("\n=== Инициализация завершена ===\n")
        return embedder, vector_db, documents, gpt_service
    except Exception as e:
        print(f"\n❌ Ошибка при инициализации: {str(e)}")
        logger.error(f"Ошибка при инициализации компонентов: {str(e)}")
        return None, None, None, None

def is_general_question(query: str) -> bool:
    general_keywords = {
        'привет', 'здравствуй', 'как дела', 'кто ты', 'что ты', 'расскажи',
        'погода', 'время', 'дата', 'день', 'месяц', 'год', 'час', 'минута',
        'секунда', 'доброе утро', 'добрый день', 'добрый вечер', 'спасибо',
        'пожалуйста', 'до свидания', 'пока', 'хорошо', 'плохо', 'отлично',
        'ужасно', 'замечательно', 'прекрасно', 'ужасно', 'плохо', 'хорошо'
    }
    
    query_words = set(query.lower().split())
    return bool(query_words & general_keywords)

def interactive_search():
    print("\n=== Запуск интерактивного режима ===")
    embedder, vector_db, documents, gpt_service = initialize_components()
    
    if not all([embedder, vector_db, documents, gpt_service]):
        print("\n❌ Не удалось инициализировать компоненты")
        logger.error("Не удалось инициализировать компоненты")
        return
    
    print("\nДобро пожаловать!")
    print("Я могу отвечать на вопросы по документам и общаться на общие темы.")
    print("Введите 'выход' для завершения работы")
    
    while True:
        try:
            query = input("\nВаш вопрос: ").strip()
            
            if query.lower() in ['выход', 'exit', 'quit']:
                print("До свидания!")
                break
                
            if not query:
                print("Пожалуйста, введите непустой вопрос")
                continue
            
            print("\nОбработка запроса...")
            if is_general_question(query):
                print("Определен общий вопрос, используем GPT...")
                response = gpt_service.generate_response(query)
                if response:
                    print("\nОтвет:")
                    print(response)
                else:
                    print("Извините, произошла ошибка при генерации ответа")
            else:
                print("Поиск в базе документов...")
                processed_query = preprocess_text(query)
                results = vector_db.search(processed_query, k=10)

                response = format_response(query, results)
                print("\nОтвет:")
                print(response)
                    
        except Exception as e:
            print(f"\n❌ Ошибка при обработке запроса: {str(e)}")
            logger.error(f"Ошибка при обработке запроса: {str(e)}")
            print("Произошла ошибка при обработке запроса. Попробуйте еще раз.")

if __name__ == "__main__":
    interactive_search() 