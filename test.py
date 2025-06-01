import sys
sys.path.insert(0, "/content/drive/MyDrive/PT-CD/OpenPrompt")

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
import torch

# 1. 클래스 및 입력 데이터 정의
classes = ["낚시성", "비낚시성"]

dataset = [
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

# 템플릿: 의미 일치 비교 중심
text='기사 제목: {"placeholder":"text_a","shortenable": True} 기사 내용: {"placeholder":"text_b"} 이 둘은 {"mask"} 관계입니다.'
# text=('기사 제목: {"placeholder":"text_a"} {"text":"</s>"} '
#         '{"text":"<s>"} 요약 내용: {"placeholder":"text_b"} {"text":"</s>"} '
#         '{"text":"<s>"}이 둘은 {"mask"} 관계입니다.')
text = (
    '기사 제목: {"placeholder":"text_a"}, '
    '기사 내용: {"placeholder":"text_b"}, '
    '이것은 가짜기사 인가요? 정답 : {"mask"}'
)

# verbalizer: 의미 판단을 돕는 단어들 사용
label_words={
    "낚시성": ["가짜"],
    "비낚시성": ["기사"],
}

# 2. 사전학습 언어모델 로드
plm, tokenizer, model_config, WrapperClass = load_plm("kobert", "skt/kobert-base-v1")

# tokenizer.add_tokens(["다른", "같은"], special_tokens=False)
# plm.resize_token_embeddings(len(tokenizer))

# 3. 프롬프트 템플릿 정의
promptTemplate = ManualTemplate(
    text=text,
    tokenizer=tokenizer,
)

label_words={
    "낚시성": ["가짜"],
    "비낚시성": ["진짜"],
}

# 4. 버벌라이저 정의 (모델 출력 -> 클래스 매핑)
promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words=label_words,
    tokenizer=tokenizer,
    multi_token_handler="first"
)
# promptVerbalizer = SoftVerbalizer(
#     model=plm,
#     classes=classes,
#     label_words=label_words,
#     tokenizer=tokenizer,
#     multi_token_handler="first"
# )
# 5. 분류용 프롬프트 모델 생성
promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)

# 6. 데이터로더 생성
data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=1,
)

# 7. 예측 실행
import torch.nn.functional as F

promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        # batch['input_ids'] shape: [1, seq_len]
        input_ids = batch['input_ids']
        mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)[0][1]  # [batch_idx, pos] → pos

        outputs = plm(input_ids=input_ids, attention_mask=batch['attention_mask'])
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        mask_logits = logits[0, mask_pos]  # [vocab_size] at [MASK] position

        topk = torch.topk(mask_logits, k=20)
        topk_ids = topk.indices.tolist()
        topk_scores = topk.values.tolist()
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids)

        print("=" * 80)
        print("Input Text:", tokenizer.decode(input_ids[0], skip_special_tokens=True))
        print(f"[MASK] 위치의 예측 상위 20개:")
        for i, (token, score) in enumerate(zip(topk_tokens, topk_scores)):
            print(f"{i+1:2d}. {token:10s}  (logit: {score:.4f})")
promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        predicted_label = classes[preds]
        print(f"Input: {input_text} → Predicted class: {predicted_label}")
