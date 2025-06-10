import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

logger = logging.getLogger(__name__)

class GPTService:
    def __init__(self, model_name: str = "ai-forever/rugpt3small_based_on_gpt2"):
        """Инициализация сервиса GPT
        
        Args:
            model_name:
        """
        try:
            logger.info(f"Загрузка модели {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise

    def generate_response(self, prompt: str, max_length: int = 200) -> Optional[str]:
        """Генерация ответа на основе промпта
        
        Args:
            prompt:
            max_length:
            
        Returns:
            Сгенерированный ответ или None в случае ошибки
        """
        try:
            full_prompt = f"Вопрос: {prompt}\nОтвет:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response = response.split("Ответ:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {str(e)}")
            return None 