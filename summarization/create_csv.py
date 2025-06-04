import os
import json
import csv
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# 요약 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

# newsContent 요약 함수
def summarize_text(text):
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

    summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return summary.strip()

# VL_Part2 제외하고 JSON 수집
def get_filtered_json_files(input_dir):
    input_dir_path = Path(input_dir).resolve()
    json_files = []
    for path in input_dir_path.rglob('*.json'):
        parts = path.relative_to(input_dir_path).parts
        if any(part.startswith("VL_Part2") for part in parts):
            continue
        json_files.append(path)
    return json_files

# 메인 처리 함수
def process_json_files(input_dir):
    output_file = 'test3.csv'

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["clickbaitClass", "title", "generatedSummary", "newsID"])  # 헤더

        json_files = get_filtered_json_files(input_dir)
        total_files = len(json_files)
        print(f'총 {total_files}개의 JSON 파일을 처리합니다...')

        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                labeled = data.get('labeledDataInfo', {})
                source = data.get('sourceDataInfo', {})

                title = labeled.get('newTitle', '').strip()
                clickbait_class = labeled.get('clickbaitClass', '')
                content = source.get('newsContent', '').strip()
                news_id = source.get('newsID', '').strip()

                if not content or not title or clickbait_class == '':
                    continue

                # summary = summarize_text(content)
                content = content.replace('\n', ' ')
                writer.writerow([clickbait_class, title, content, news_id])

                if i % 100 == 0:
                    print(f'진행률: {i}/{total_files} 파일 처리 완료')

            except Exception as e:
                print(f'파일 {json_file} 처리 중 오류 발생: {str(e)}')
                continue

    print(f'\n처리가 완료되었습니다. 결과가 {output_file}에 저장되었습니다.')

# 사용자 입력 받는 main 함수
def main():
    while True:
        input_dir = input('JSON 파일이 있는 디렉토리 경로를 입력하세요: ').strip()
        if os.path.isdir(input_dir):
            break
        else:
            print('유효하지 않은 디렉토리 경로입니다. 다시 시도해주세요.')

    process_json_files(input_dir)

if __name__ == '__main__':
    main()
