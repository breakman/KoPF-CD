import csv
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# KoBART 요약 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

def summarize(text):
    text = text.replace('\n', ' ').replace('\\', '').replace('\"', '"')
    input_ids = tokenizer.encode(
        text,
        max_length=1024,
        truncation=True,
        return_tensors='pt'
    )

    summary_ids = model.generate(
        input_ids,
        max_length=128,
        min_length=32,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True).strip()

def summarize_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

        for i, row in enumerate(reader, 1):
            if len(row) != 4:
                print(f"[스킵] {i}번째 줄은 4개의 필드가 아님: {row}")
                continue

            label, title, content, news_id = row
            try:
                summary = summarize(content)
                writer.writerow([label, title, summary, news_id])
            except Exception as e:
                print(f"[오류] {i}번째 줄 요약 실패: {str(e)}")
                continue

            if i % 10 == 0:
                print(f"{i}개 기사 요약 완료")

    print(f"요약 완료 → 저장 위치: {output_path}")

if __name__ == "__main__":
    input_csv = "balanced_sample.csv"       # 너가 준 원본 CSV
    output_csv = "summarized_articles.csv" # 결과 저장 파일
    summarize_csv(input_csv, output_csv)
