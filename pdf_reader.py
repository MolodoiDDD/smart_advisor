import PyPDF2
import os
import re

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def find_definition(text):
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_lower = sentence.lower()
        if 'стипендия' in sentence_lower and any([
            'это' in sentence_lower,
            'является' in sentence_lower,
            'представляет' in sentence_lower,
            'денежная выплата' in sentence_lower,
            'выплачивается' in sentence_lower,
            'назначается' in sentence_lower,
            'предоставляется' in sentence_lower,
            'материальная поддержка' in sentence_lower,
            'материальная помощь' in sentence_lower,
            'денежное пособие' in sentence_lower
        ]):
            return sentence

def main():
    pdf_dir = "F:/VKR/download/data/pdf"
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"\nОбработка файла: {filename}")
            try:
                text = extract_text_from_pdf(pdf_path)
                definition = find_definition(text)
                if definition:
                    print(f"Найдено определение в файле {filename}:")
                    print(definition)
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {str(e)}")

if __name__ == "__main__":
    main() 