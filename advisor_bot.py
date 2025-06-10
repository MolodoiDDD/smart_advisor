import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

from smart_advisor.core.embeddings import EmbeddingService
from smart_advisor.core.database import VectorDatabase
from smart_advisor.services.query_processor import QueryProcessor
from smart_advisor.config.config import config

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ScholarshipBot:
    def __init__(self, token: str):
        self.token = token
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDatabase(self.embedding_service)
        self.query_processor = QueryProcessor(self.vector_db)
        
    def create_main_menu(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("❓ Помощь", callback_data="help"),
                InlineKeyboardButton("ℹ️ О боте", callback_data="about")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_text = (
            "👋 Добро пожаловать в систему умного советника!\n\n"
            "Я помогу вам получить информацию о стипендиях и других аспектах студенческой жизни.\n\n"
            "Вы можете задать вопросы о стипендиях, например:\n"
            "- Что такое стипендия?\n"
            "- Сколько составляет обычная стипендия?\n"
            "- Какие требования для получения повышенной стипендии?\n"
            "- Когда выплачивается стипендия?\n"
            "- Как оформить социальную стипендию?\n\n"
            "Используйте меню ниже для навигации или просто задайте свой вопрос."
        )
        await update.message.reply_text(welcome_text, reply_markup=self.create_main_menu())

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "📚 Справка по использованию бота:\n\n"
            "1. Просто отправьте свой вопрос о стипендиях, и я постараюсь на него ответить.\n\n"
            "2. Примеры вопросов:\n"
            "   - Что такое стипендия?\n"
            "   - Сколько составляет обычная стипендия?\n"
            "   - Какие требования для получения повышенной стипендии?\n"
            "   - Когда выплачивается стипендия?\n"
            "   - Как оформить социальную стипендию?\n\n"
            "3. Доступные команды:\n"
            "   /start - Начать работу с ботом\n"
            "   /help - Показать эту справку\n"
            "   /about - Информация о боте\n"
            "   /feedback - Отправить обратную связь\n\n"
            "4. Используйте меню для быстрой навигации."
        )
        await update.message.reply_text(help_text, reply_markup=self.create_main_menu())

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            processing_message = await update.message.reply_text(
                "⏳ Обрабатываю ваш запрос...",
                reply_markup=self.create_main_menu()
            )
            
            response = self.query_processor.process_query(update.message.text)
            
            if hasattr(response, 'answer'):
                answer = response.answer
                confidence = response.confidence
                
                if confidence > 0.7:
                    confidence_text = "✅ Уверенность в ответе: Высокая"
                elif confidence > 0.4:
                    confidence_text = "⚠️ Уверенность в ответе: Средняя"
                else:
                    confidence_text = "❌ Уверенность в ответе: Низкая"
                
                await processing_message.edit_text(
                    f"{answer}\n\n{confidence_text}",
                    reply_markup=self.create_main_menu()
                )
            else:
                await processing_message.edit_text(
                    str(response),
                    reply_markup=self.create_main_menu()
                )
                
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {str(e)}")
            await update.message.reply_text(
                "😔 Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте переформулировать вопрос.",
                reply_markup=self.create_main_menu()
            )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        callback_data = query.data
        
        if callback_data == "help":
            text = (
                "❓ Помощь по использованию бота:\n\n"
                "1. Задавайте вопросы о стипендиях\n"
                "2. Используйте команды:\n"
                "   /start - Начать работу\n"
                "   /help - Показать справку\n"
                "   /about - О боте"
            )
        elif callback_data == "about":
            text = (
                "ℹ️ О боте:\n\n"
                "Smart Advisor - умный помощник для студентов.\n"
                "Версия: 1.0.0\n"
                "Последнее обновление: 21.05.2024"
            )
        else:
            text = "Пожалуйста, выберите действие из меню."
        
        await query.message.edit_text(text, reply_markup=self.create_main_menu())

    def run(self):
        application = Application.builder().token(self.token).build()

        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.add_handler(CallbackQueryHandler(self.handle_callback))

        application.run_polling()

def main():
    bot = ScholarshipBot(config.TELEGRAM_BOT_TOKEN)
    bot.run()

if __name__ == '__main__':
    main() 