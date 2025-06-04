import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

text = "뉴미디어기반 콘텐츠제작업체 마젠타컴퍼니는 오는 16일 독일 라인란트 팔츠 주립 필하모니(Staatsphilharmonie Rheinland-Pfalz)의 비대면 VR라이브 오케스트라공연(독일 공연명 : 360° 라이브 스트림)을 한국에서 원격으로 중계한다. 원격으로 중계가 가능한 것은 마젠타 컴퍼니가 자체개발한 5G VR 로봇 '아이디어'(I-DeaR·Intelligent Digital Robotics) 덕분이다. 지난해 마젠타컴퍼니는 과학기술정보통신부와 한국전파진흥협회 국제공동 사업지원으로 '아이디어'라는 초고화질(4K급) VR 촬영 로봇을 개발했다. 5G망으로 국가 간 원격 조정이 가능한 모빌리티 플랫폼(RC카 형태)이다. 촬영장비가 어디에 있던 LTE급 이상 통신 상태면 원격 조정 및 영상전송이 가능하다. 코로나19로 인한 비대면 사회에 최적화된 기술로 공연장이나 여행지를 360도 VR 영상 4K로 촬영해 5G 스마트폰을 이용해 전송한다. 고화질을 빠르게 전송하기 위해 4K 영상을 4분할해 전송한 뒤 결합하는 기술을 개발해 적용했다. 마젠타 컴퍼니는 독일 라인란트 팔츠 주립 필하모니의 공식 초청을 받고 오는 16일 한국 시각 오후 5시(독일 현지시각 오전 10시)부터 약 1시간 동안 공연 및 실시간 송출할 예정이다. 한국전파진흥협회 관계자는 미디어SR에 현재 5G VR 로봇 '아이디어'는 독일 현지에 가 있는 상황이라며 독일 현지 스탭과 협력해 한국에서 원격으로 공연을 중계하게 될 것이라고 귀띔했다. 공연 실황은 독일 라인란트 팔츠 주립 필하모니 공식 유튜브 채널과 마젠타 컴퍼니 I-DeaR 유튜브 채널, 국내 SKT jumpVR(VR전용 OTT플랫폼)을 통해 송출한다. 국내 클래식 애호가 등이 온라인상에서 양방향 소통할 수 있도록 지휘자와의 대화 등도 준비 중이다. 독일 라인란트 팔츠 주립 필하모니는 직접 공연관람이 힘든 지금, 다양한 방법으로 실감 공연을 즐길 수 있도록 '아이디어'를 초청했으며, 오케스트라 측에서 운송비, 스탭 체제비용 등 1만 유로상당 현금을 지원했다. '아이디어'가 연주 전 상황과 무대 위를 주행할 수 있도록 파격적 동선도 허용했다. '아이디어'는 독일 하이델베르크시, 폴란드 우츠 시, 이탈리아 관광청, 멕시코 시티, 브라질 시티 등과 업무협약을 맺고 비대면 원격 여행도 준비하고 있다. 한편 해당 기관의 전용 유튜브 채널과 마젠타 컴퍼니 I-DeaR 유튜브 채널로 실시간 송출될 예정이다."

text = text.replace('\n', ' ')
# Tokenize with truncation
input_ids = tokenizer.encode(
    text,
    max_length=1024,
    truncation=True,
    return_tensors='pt'
)

# Generate summary with better decoding parameters
summary_ids = model.generate(
    input_ids,
    max_length=128,
    min_length=32,
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True
)

text2 = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

print(text2)

