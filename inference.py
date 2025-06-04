import sys
sys.path.insert(0, "/content/drive/MyDrive/PT-CD/OpenPrompt")

import torch
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, SoftVerbalizer
from openprompt import PromptForClassification, PromptDataLoader

# 1. 분류 클래스 정의
classes = ["낚시성", "비낚시성"]
label_words = {
    "낚시성": ["조작"],
    "비낚시성": ["진짜"]
}

# 2. 템플릿 정의 (학습에 사용한 것과 동일해야 함)
template_text = '제목과 내용을 포함한 낚시성 뉴스 항목 {"mask"}입니다: {"placeholder": "text_a"}, {"placeholder": "text_b"}'

# 3. 사전학습 언어모델 및 토크나이저 로드
plm, tokenizer, model_config, WrapperClass = load_plm("xlmroberta", "FacebookAI/xlm-roberta-base")

# 4. 템플릿 구성
prompt_template = ManualTemplate(
    text=template_text,
    tokenizer=tokenizer,
)

# 5. 버벌라이저 구성
prompt_verbalizer = SoftVerbalizer(
    model=plm,
    classes=classes,
    label_words=label_words,
    tokenizer=tokenizer,
    multi_token_handler="first"
)

# 6. 프롬프트 분류 모델 생성
prompt_model = PromptForClassification(
    template=prompt_template,
    plm=plm,
    verbalizer=prompt_verbalizer,
)

# 7. 저장된 학습된 weight 로드
ckpt_path = "/content/template3.ckpt"  # 학습시 저장한 .ckpt 경로
prompt_model.load_state_dict(torch.load(ckpt_path))
prompt_model = prompt_model.cuda()
prompt_model.eval()

# 8. 추론용 입력 예시 (InputExample list)
examples = [
    InputExample(
        guid=0,
        text_a="연구개발특구 '소통의 장' 연다",
        text_b="국세청 100억 이하 기업 정기조사서 제외",
    ),
    InputExample(
        guid=1,
        text_a="대덕특구 출연연 기관장 평균재산은 17억원",
        text_b="오이·배추 등 두달만에 28개 품목 인하",
    ),
    InputExample(
        guid=2,
        text_a="美빅텍에 '노조 바람'… 구글 이어 아마존서도 결성 움직임",
        text_b="앨러배마주 창고 직원 수천명, 노조 설립 우편투표 중'무노조 경영' 고수해온 아마존, 창사 이래 가장 큰 도전",
    ),
    InputExample(
        guid=3,
        text_a="[단독] 송희영 재판도 인정한 '안종범 업무수첩'",
        text_b="[송희영 재판 판결문①] 업무수첩 통해 김기춘 대우조선 인사개입 정황…송희영·안종범 연결고리는 조선일보 靑 출입기자",
    ),
    InputExample(
        guid=4,
        text_a="2024년도 대학 입시, 세부 능력 및 특기 사항 관리 어떻게 해야 하나?",
        text_b="'수업 집중' 기본 지키고···수행평가서 역량 최대한 보여줘야",
    ),
    InputExample(
        guid=5,
        text_a="하이트진로, 매실주 브랜드 '매화수' 새단장",
        text_b="하이트진로는 주류시장의 패키지 디자인 다변화와 저도화 트렌드를 반영해 매화수를 리뉴얼하고 시장확대에 나선다고 밝혔다.",
    ),
    InputExample(
        guid=6,
        text_a="[단독] 송희영 재판도 인정한 '안종범 업무수첩'",
        text_b="하이트진로는 주류시장의 패키지 디자인 다변화와 저도화 트렌드를 반영해 매화수를 리뉴얼하고 시장확대에 나선다고 밝혔다.",
    ),
    
]

# 9. 데이터로더 생성
data_loader = PromptDataLoader(
    dataset=examples,
    tokenizer=tokenizer,
    template=prompt_template,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=1,
    shuffle=False
)

# 10. 예측 실행
with torch.no_grad():
    for batch in data_loader:
        batch = batch.cuda()
        logits = prompt_model(batch)
        preds = torch.argmax(logits, dim=-1)
        label = classes[preds.item()]
        input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        print("=" * 80)
        print(f"입력 문장: {input_text}")
        print(f"예측 결과: {label}")
