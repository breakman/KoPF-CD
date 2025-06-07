import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# KoBART 요약 모델 로드 (앱 시작 시 한 번만 로드되도록 설정)
_tokenizer_summarize = None
_model_summarize = None
_device_summarize = None

def init_summarizer():
    global _tokenizer_summarize, _model_summarize, _device_summarize
    if _tokenizer_summarize is None: # 중복 로딩 방지
        _tokenizer_summarize = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
        _device_summarize = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model_summarize = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization').to(_device_summarize)
        _model_summarize.eval() # 추론 모드로 설정
        print("Summarization model loaded.")

def summarize_text(text: str) -> str:
    if _tokenizer_summarize is None or _model_summarize is None:
        raise Exception("Summarizer not initialized. Call init_summarizer() first.")

    text = text.replace('\n', ' ').replace('\\', '').replace('\"', '"') # 개행 및 불필요한 백슬래시, 따옴표 처리
    
    input_ids = _tokenizer_summarize.encode(
        text,
        max_length=1024,
        truncation=True,
        return_tensors='pt'
    ).to(_device_summarize)

    summary_ids = _model_summarize.generate(
        input_ids,
        max_length=128, 
        min_length=32,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    return _tokenizer_summarize.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True).strip()

# 예시: CSV 처리 부분은 웹 앱에서는 직접 사용하지 않으므로 제거하거나 주석 처리합니다.
# def summarize_csv(input_path, output_path):
# ... (이하 생략)

# if __name__ == "__main__":
# init_summarizer()
# sample_text = "여기에 테스트할 긴 텍스트를 넣어주세요. 이 텍스트를 요약 모델이 잘 처리하는지 확인해봅시다."
# summary = summarize_text(sample_text)
# print(f"Original Text: {sample_text}")
# print(f"Summary: {summary}") 