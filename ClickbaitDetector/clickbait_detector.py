import torch
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, SoftVerbalizer
from openprompt import PromptForClassification, PromptDataLoader

# 1. 분류 클래스 정의
_classes_clickbait = ["낚시성", "비낚시성"]
_label_words_clickbait = {
    "낚시성": ["참"],
    "비낚시성": ["잘못"]
}

# 2. 템플릿 정의
_template_text_clickbait = '제목과 내용을 포함한 낚시성 뉴스 항목 {"mask"}입니다: {"placeholder": "text_a"}, {"placeholder": "text_b"}'

_plm_clickbait = None
_tokenizer_clickbait = None
_model_config_clickbait = None
_WrapperClass_clickbait = None
_prompt_model_clickbait = None
_device_clickbait = None

def init_clickbait_detector(ckpt_path: str):
    global _plm_clickbait, _tokenizer_clickbait, _model_config_clickbait, _WrapperClass_clickbait, _prompt_model_clickbait, _device_clickbait

    if _prompt_model_clickbait is not None: # 중복 로딩 방지
        print("Clickbait detector model already loaded.")
        return

    _device_clickbait = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 사전학습 언어모델 및 토크나이저 로드
    _plm_clickbait, _tokenizer_clickbait, _model_config_clickbait, _WrapperClass_clickbait = load_plm("xlmroberta", "FacebookAI/xlm-roberta-base")

    # 4. 템플릿 구성
    prompt_template = ManualTemplate(
        text=_template_text_clickbait,
        tokenizer=_tokenizer_clickbait,
    )

    # 5. 버벌라이저 구성
    prompt_verbalizer = SoftVerbalizer(
        model=_plm_clickbait,
        classes=_classes_clickbait,
        label_words=_label_words_clickbait,
        tokenizer=_tokenizer_clickbait,
        multi_token_handler="first"
    )

    # 6. 프롬프트 분류 모델 생성
    _prompt_model_clickbait = PromptForClassification(
        template=prompt_template,
        plm=_plm_clickbait,
        verbalizer=prompt_verbalizer,
    )

    # 7. 저장된 학습된 weight 로드
    # ckpt_path는 실제 가중치 파일 경로로 수정 필요
    try:
        _prompt_model_clickbait.load_state_dict(torch.load(ckpt_path, map_location=_device_clickbait))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}. Please provide the correct path.")
        # 혹은 여기서 기본 가중치를 로드하거나, 에러를 발생시켜 앱 실행을 중단할 수 있습니다.
        # 여기서는 일단 모델 객체는 생성된 상태로 두되, 예측 시 오류가 발생하도록 합니다.
        _prompt_model_clickbait = None # 로드 실패 시 모델 사용 불가
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        _prompt_model_clickbait = None
        raise
        
    _prompt_model_clickbait = _prompt_model_clickbait.to(_device_clickbait)
    _prompt_model_clickbait.eval()
    print("Clickbait detector model loaded.")

def predict_clickbait(title: str, content: str) -> str:
    if _prompt_model_clickbait is None or _tokenizer_clickbait is None or _WrapperClass_clickbait is None:
        raise Exception("Clickbait detector not initialized or model loading failed. Call init_clickbait_detector() first with a valid ckpt_path.")

    example = [InputExample(guid=0, text_a=title, text_b=content)]
    
    data_loader = PromptDataLoader(
        dataset=example,
        tokenizer=_tokenizer_clickbait,
        template=_prompt_model_clickbait.template, # 여기서 template을 가져옵니다.
        tokenizer_wrapper_class=_WrapperClass_clickbait,
        batch_size=1,
        shuffle=False
    )

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(_device_clickbait)
            logits = _prompt_model_clickbait(batch)
            preds = torch.argmax(logits, dim=-1)
            return _classes_clickbait[preds.item()]
    return "판단 불가"

# if __name__ == '__main__':
    # 실제 .ckpt 파일 경로로 수정해야 합니다.
    # 예: CKPT_FILE_PATH = 'path/to/your/template3.ckpt'
    # init_clickbait_detector(CKPT_FILE_PATH)
    
    # if _prompt_model_clickbait:
        # test_title = "[단독] 놀라운 사실이 밝혀졌다!"
        # test_content = "이 기사를 클릭하지 않으면 후회할 것입니다. 하지만 실제 내용은 별것 없습니다."
        # prediction = predict_clickbait(test_title, test_content)
        # print(f"Title: {test_title}")
        # print(f"Content: {test_content}")
        # print(f"Prediction: {prediction}")
        
        # test_title_2 = "정부, 내년 경제 성장률 2.5% 전망"
        # test_content_2 = "기획재정부는 내년도 경제정책방향을 발표하며, 수출 회복과 내수 진작을 통해 2.5% 경제 성장을 목표로 한다고 밝혔다."
        # prediction_2 = predict_clickbait(test_title_2, test_content_2)
        # print(f"\nTitle: {test_title_2}")
        # print(f"Content: {test_content_2}")
        # print(f"Prediction: {prediction_2}") 