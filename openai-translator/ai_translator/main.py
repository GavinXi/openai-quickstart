import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, ConfigLoader
from model import OpenAIModel
from translator import PDFTranslator

if __name__ == "__main__":
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()
    config_loader = ConfigLoader(args.config)

    config = config_loader.load_config()

    model_name = args.openai_model if args.openai_model else config['OpenAIModel']['model']
    api_key = args.openai_api_key if args.openai_api_key else config['OpenAIModel']['api_key']
    base_url = args.openai_base_url if args.openai_base_url else config['OpenAIModel']['base_url']
    language = args.target_language if args.target_language else "Chinese"
    model = OpenAIModel(model=model_name, api_key=api_key, base_url=base_url)

    pdf_file_path = args.book if args.book else config['common']['book']
    file_format = args.file_format if args.file_format else config['common']['file_format']

    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    translator = PDFTranslator(model)
    # translator.translate_pdf(pdf_file_path, file_format)
    language_map = {
        "Vietnamese": "越南语",
        "Chinese": "中文",
        "Korean": "韩文",
        "Japanese": "日文"
    }

    if language in language_map:
        translated_language = language_map[language]
        output_file_path = f"../tests/{language}_translated.md"
        translator.translate_pdf(pdf_file_path, file_format, target_language=translated_language,
                                 output_file_path=output_file_path)
