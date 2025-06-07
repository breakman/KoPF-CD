import os
from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from ClickbaitDetector.summarization_utils import init_summarizer, summarize_text
from ClickbaitDetector.clickbait_detector import init_clickbait_detector, predict_clickbait
import re # 정규표현식 모듈 추가
# from requests_html import HTMLSession # requests-html 임포트 제거
# import asyncio # asyncio 임포트 제거
from selenium import webdriver # Selenium 임포트 추가
import time # time 임포트 추가

app = Flask(__name__)

# --- CONFIGURATION ---
# 실제 template3.ckpt 파일의 경로로 수정해주세요.
# 예: 'path/to/your/template3.ckpt' 또는 만약 app.py와 같은 디렉토리에 있다면 'template3.ckpt'
CLICKBAIT_MODEL_CKPT_PATH = './ckpts/8219.ckpt' 

# --- MODEL INITIALIZATION ---
try:
    print("Initializing models...")
    init_summarizer()
    # OpenPrompt는 특정 경로에 의존할 수 있으므로, 필요시 sys.path 조작이 필요할 수 있습니다.
    # 예: import sys; sys.path.insert(0, "/path/to/OpenPrompt")
    # clickbait_detector.py 내에서 OpenPrompt 라이브러리를 찾을 수 있도록 환경 설정이 중요합니다.
    init_clickbait_detector(CLICKBAIT_MODEL_CKPT_PATH)
    print("Models initialized successfully.")
except Exception as e:
    print(f"Error during model initialization: {e}")
    # 프로덕션 환경에서는 여기서 앱 실행을 중단하거나, 
    # 모델 없이 실행되는 폴백(fallback) 로직을 고려해야 합니다.

def fetch_article_content(url: str):
    """주어진 URL에서 기사 제목과 본문을 추출합니다."""
    title = "제목을 찾을 수 없습니다." # 함수 초기에 기본값 설정
    content = "본문 내용을 찾을 수 없습니다." # 함수 초기에 기본값 설정
    # session = None # HTMLSession 객체를 위한 변수 제거
    driver = None # Selenium WebDriver 객체를 위한 변수

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        html_content_for_soup = "" # BeautifulSoup에 전달될 HTML 컨텐츠 (str 또는 bytes)
        final_encoding = 'utf-8' # 기본 인코딩, requests 사용 시 재결정됨

        if 'chosun.com' in url.lower():
            print(f"Using Selenium for Chosun URL: {url}")
            try:
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")  # 브라우저 창 안 띄움 (qwe.py 참조)
                options.add_argument("--no-sandbox") # Linux 환경에서 headless 실행 시 종종 필요
                options.add_argument("--disable-dev-shm-usage") # Docker 또는 CI 환경에서 안정성 향상
                options.add_argument("--disable-gpu") # 일부 headless 환경에서 도움될 수 있음
                # ChromeDriver가 시스템 PATH에 없거나 특정 드라이버를 사용해야 하는 경우, 아래처럼 경로를 지정할 수 있습니다.
                # 예: driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver', options=options)
                # 또는 webdriver_manager 같은 라이브러리를 사용하여 드라이버 관리를 자동화할 수도 있습니다.
                driver = webdriver.Chrome(options=options)
                driver.get(url)
                
                # JS 로딩 대기
                print(f"Waiting for JS to load on {url} (1 seconds)...")
                time.sleep(1) 
                
                html_content_for_soup = driver.page_source # Selenium으로부터 페이지 소스 가져오기 (문자열)
                print(f"Chosun page loaded via Selenium. HTML length: {len(html_content_for_soup)}")
                # Selenium 사용 시 final_encoding은 BeautifulSoup이 문자열을 직접 받으므로 크게 중요하지 않음.
                # BeautifulSoup은 내부적으로 UTF-8로 처리하려고 시도함.
            except Exception as selenium_e:
                print(f"Selenium error occurred for {url}: {selenium_e}")
                # Selenium 오류 발생 시, title과 content는 기본 오류 메시지로 유지하고 finally에서 driver.quit() 실행
                # 혹은 여기서 특정 오류 값 반환 후 predict_route에서 처리하도록 할 수 있음
                # 여기서는 기존 구조를 따라 예외를 다시 발생시켜 외부 try-except에서 처리하도록 함
                raise selenium_e 
        else:
            print(f"Using standard requests for URL: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # 오류 발생 시 예외 발생
            
            # --- 인코딩 처리 강화 시작 (기존 로직 유지) ---
            content_type_header = response.headers.get('Content-Type', '').lower()
            print(f"Original Content-Type header for {url}: {content_type_header}")
            print(f"Requests initial response.encoding for {url}: {response.encoding}")

            # final_encoding은 위에서 'utf-8'로 초기화됨
            
            # 1. Content-Type 헤더에서 charset 파싱 시도
            if 'charset=' in content_type_header:
                parsed_encoding = content_type_header.split('charset=')[-1].split(';')[0].strip()
                if parsed_encoding: # 실제 파싱된 값이 있을 경우에만 사용
                    final_encoding = parsed_encoding
                print(f"Encoding from Content-Type header for {url}: {final_encoding}")

            # 2. 특정 URL (서울신문)에 대해 UTF-8 강제 (더 높은 우선순위)
            if 'seoul.co.kr' in url.lower():
                print(f"Seoul.co.kr detected, ensuring UTF-8 for {url}.")
                final_encoding = 'utf-8'
            
            # 3. Requests의 apparent_encoding 사용 (Content-Type에 없거나, 특정 URL이 아닐 때)
            if not final_encoding or final_encoding.lower() == 'iso-8859-1': # ISO-8859-1은 종종 잘못된 추측
                apparent_enc = response.apparent_encoding
                print(f"Requests response.apparent_encoding for {url}: {apparent_enc}")
                if apparent_enc and apparent_enc.lower() != 'iso-8859-1':
                    final_encoding = apparent_enc
                else: # apparent_encoding도 없거나 ISO-8859-1이면 UTF-8로 기본 설정
                    final_encoding = 'utf-8' # 이미 기본값이지만 명시
                    print(f"Defaulting to UTF-8 for {url} as apparent_encoding is {apparent_enc} or final_encoding was {final_encoding}")
            
            print(f"Final encoding chosen for BeautifulSoup for {url}: {final_encoding}")
            
            current_content_bytes = response.content
            if final_encoding and final_encoding.lower() == 'utf-8' and current_content_bytes.startswith(b'\xef\xbb\xbf'):
                print(f"UTF-8 BOM detected and removed for {url}.")
                current_content_bytes = current_content_bytes[3:]
            html_content_for_soup = current_content_bytes # bytes 객체
            # --- 인코딩 처리 강화 끝 ---
        
        # BeautifulSoup 객체 생성
        # html_content_for_soup가 bytes이면 from_encoding 사용, str이면 (Selenium 결과) 그대로 파서에 전달
        if isinstance(html_content_for_soup, bytes):
            soup = BeautifulSoup(html_content_for_soup, 'html.parser', from_encoding=final_encoding)
        else: # str (주로 Selenium 결과)
            soup = BeautifulSoup(html_content_for_soup, 'html.parser')

        # title, content 변수는 함수 시작 시 이미 기본값으로 초기화됨.
        # 각 사이트별 파서는 title 또는 content가 기본값일 경우에만 값을 채우려고 시도.

        # -1. 엑스포츠뉴스 시도
        if title == "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip()):
            title_tag_xports = soup.select_one('h1.at_title')
            if title_tag_xports and title == "제목을 찾을 수 없습니다.": # 아직 제목 못찾았을때만
                title = title_tag_xports.get_text(strip=True)
            
            content_div_xports = soup.select_one('div.news_contents')
            # 제목을 찾았고, 아직 본문을 못 찾았거나 내용이 없을때만 시도 (또는 제목 아직 못찾았지만 일단 본문 파서 시도)
            if content_div_xports and (title != "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip())): 
                for unwanted_tag in content_div_xports.select(
                    'script, style, iframe, .ad, .tb.top, .another_ats, .keyword_zone, #fb-root, .sns_area, .rec_at, .cmt_zone, #taboola-below-article-thumbnails-desktop, .video_frm, .photobox, .iwmads, .pop_pr_info'
                ):
                    unwanted_tag.decompose()

                for img_tag in content_div_xports.find_all('img'):
                    img_tag.decompose()
                
                for br_tag in content_div_xports.find_all("br"):
                    br_tag.replace_with('\n') 
                
                full_text = content_div_xports.get_text(separator='\n')
                
                temp_lines = []
                for line_content in full_text.splitlines():
                    stripped_line = line_content.strip()
                    if stripped_line:
                        temp_lines.append(stripped_line)
                
                if not temp_lines:
                    # content는 기본값("본문 내용을 찾을 수 없습니다.") 유지
                    pass 
                else:
                    collected_content = []
                    lines_to_process = temp_lines

                    if temp_lines: # 첫 줄에서 기자 정보 처리
                        first_line = temp_lines[0]
                        match_reporter = re.match(r"^\s*\(\s*엑스포츠뉴스\s*[^\)]+\s*기자\s*\)\s*(.*)", first_line)
                        if match_reporter:
                            text_after_reporter = match_reporter.group(1).strip()
                            if text_after_reporter:
                                collected_content.append(text_after_reporter)
                            lines_to_process = temp_lines[1:] # 다음 줄부터 처리
                        # (기자 이름) 기자 = 패턴은 첫 줄에서만 특별히 처리하지 않고, 아래 반복문에서 일반 텍스트로 포함될 수 있도록 함

                    for line in lines_to_process:
                        if "사진=" in line or \
                           re.search(r"^[^\s]+\s*기자\s+.*@xportsnews\.com", line) or \
                           "ⓒ 엑스포츠뉴스" in line or \
                           "무단전재 및 재배포 금지" in line or \
                           line.lower().startswith("기사제공") or \
                           line.count('#') > 2:
                            break
                        collected_content.append(line)
                    
                    if collected_content and (content == "본문 내용을 찾을 수 없습니다." or not content.strip()): # 본문이 비어있을때만 덮어쓰기
                        content = '\n'.join(collected_content).strip()
                    # else: content는 기본값 또는 이전 사이트에서 찾은 값 유지 (이 조건문은 엑스포츠가 첫 파서이므로 사실상 항상 참)

        # 0. 서울신문 시도
        if title == "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip()):
            title_tag_seoul = soup.select_one('.viewHeader h1.h38')
            if not title_tag_seoul:
                title_tag_seoul = soup.select_one('.viewHeader-wrapper h1')

            if title_tag_seoul and title == "제목을 찾을 수 없습니다.":
                title = title_tag_seoul.get_text(strip=True)
            
            content_div_seoul = soup.select_one('div#articleContent div.viewContent')
            if content_div_seoul and (title != "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip())):
                for unwanted_tag in content_div_seoul.select(
                    'script, style, iframe, .v_photoarea_R, .v_photoarea_L, .v_photoarea, figure, .article_photo_left, .article_photo_right, '
                    '.ad-template, .iwmads, .byline, .articleCopyright, .bottom_banner, .social_widget, .article_relation, '
                    '.modal, .expendImageWrap, .view_photobox, .photogallery, .vod_area, .txt_caption, .img_desc, .view_related_article, .news_date, .news_like_box, '
                    'div[style*="text-align:center;margin:40px auto 20px;"]' # 중간 광고 영역 div
                ):
                    unwanted_tag.decompose()
                
                for br_tag in content_div_seoul.find_all("br"):
                    br_tag.replace_with('\n') 
                
                processed_texts = []
                
                h2_elements = content_div_seoul.find_all('h2', class_='stit')
                for h2 in h2_elements:
                    h2_text = h2.get_text(strip=True)
                    if h2_text:
                        processed_texts.append(f"--- {h2_text} ---")
                    h2.decompose()

                strong_elements = content_div_seoul.find_all('strong')
                for strong in strong_elements:
                    strong_text = strong.get_text(strip=True)
                    if strong_text.startswith("●"):
                        processed_texts.append(strong_text)
                        if strong.parent.name in ['p', 'div'] and len(strong.parent.get_text(strip=True)) == len(strong_text):
                            strong.parent.decompose()
                        else:
                            strong.decompose()
                
                # <br> 태그가 strong/h2 요소 제거 후 새로 노출될 수 있으므로 한번 더 처리
                for br_tag in content_div_seoul.find_all("br"): 
                    br_tag.replace_with('\n')

                main_content_text = content_div_seoul.get_text(separator='\n', strip=True)
                
                if main_content_text:
                    # 이미 처리된 부제목 등이 중복 포함되는 것을 방지
                    for h2_text_part in processed_texts: 
                        if h2_text_part.startswith("--- ") and h2_text_part.endswith(" ---"):
                            actual_h2_text = h2_text_part[4:-4]
                            main_content_text = main_content_text.replace(actual_h2_text, "")
                    processed_texts.append(main_content_text)
                
                temp_content_seoul = ""
                if processed_texts:
                    temp_content_seoul = '\n'.join(processed_texts).strip()
                
                if temp_content_seoul and (content == "본문 내용을 찾을 수 없습니다." or not content.strip()): # 본문이 비어있을때만 덮어쓰기
                    lines = [line.strip() for line in temp_content_seoul.splitlines()]
                    meaningful_lines = [line for line in lines if line]
                    
                    # 바이라인 제거 로직 (기존 유지)
                    lines_to_remove_at_end = 0
                    for i in range(len(meaningful_lines) - 1, max(-1, len(meaningful_lines) - 5), -1): # 마지막 4줄 정도 확인
                        line_lower = meaningful_lines[i].lower()
                        is_byline_candidate = ("기자" in line_lower or \
                                              "@" in line_lower or \
                                              "yna.co.kr" in line_lower or \
                                              "edaily.co.kr" in line_lower or \
                                              "seoul.co.kr" in line_lower or \
                                              "news@" in line_lower or \
                                              "mailto:" in line_lower or \
                                              "온라인뉴스부" in meaningful_lines[i] or \
                                              "ⓒ" in meaningful_lines[i] or \
                                              "copyright" in line_lower)

                        if is_byline_candidate and len(meaningful_lines[i]) < 100: # 너무 길지 않은 라인
                            lines_to_remove_at_end += 1
                        elif lines_to_remove_at_end > 0: # 바이라인 후보가 더 이상 아니면 중단
                            break
                        elif len(meaningful_lines[i]) > 150 or not is_byline_candidate: 
                             break

                    if lines_to_remove_at_end > 0:
                        meaningful_lines = meaningful_lines[:-lines_to_remove_at_end]
                    
                    content = '\n'.join(meaningful_lines)


        # 1. 이데일리 시도
        if title == "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip()):
            title_tag_edaily = soup.select_one('.news_titles h1')
            if title_tag_edaily and title == "제목을 찾을 수 없습니다.":
                title = title_tag_edaily.get_text(strip=True)
        
            content_div_edaily = soup.find('div', class_='news_body')
            if content_div_edaily and (title != "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip())):
                for unwanted_tag in content_div_edaily.select('script, style, iframe, .gg_textshow, .view_ad01, .view_ad02, .reporter_info, .relation_news, .to_top, .news_like_area, .copyright'):
                    unwanted_tag.decompose()
                for br in content_div_edaily.find_all("br"):
                    br.replace_with('\n') 
                content_texts = []
                # 이데일리 본문 p 태그 위주로 가져오도록 수정
                paragraphs = content_div_edaily.find_all('p', recursive=True)
                if paragraphs:
                    for p_tag in paragraphs:
                        p_text = p_tag.get_text(separator='\n', strip=True)
                        if p_text:
                            content_texts.append(p_text)
                
                if not content_texts or "".join(content_texts).strip() == "": # p태그에서 못찾았으면 전체 텍스트 시도
                    all_text = content_div_edaily.get_text(separator='\n', strip=True) 
                    if all_text.strip(): content_texts = [all_text]

                if content_texts and (content == "본문 내용을 찾을 수 없습니다." or not content.strip()): # 본문이 비어있을때만 덮어쓰기
                    content = '\n'.join(content_texts).strip() 


        # 1.5 조선일보 시도 (Selenium으로 가져온 soup 사용, 기존 BeautifulSoup 파싱 로직과 주석 유지)
        if title == "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip()):
            # 조선일보는 위에서 Selenium으로 JS 렌더링된 page_source를 soup으로 변환하여 사용.
            # 따라서, 여기서는 soup 객체를 사용하여 기존의 조선일보 파싱 로직을 그대로 적용.
            if 'chosun.com' in url.lower(): # 조선일보 URL인 경우에만 이 파서 로직 실행 (중복 방지 및 명확화)
                print("Chosun Parser (Post-Selenium): Attempting to parse with BeautifulSoup.")
            
            # 제목 추출 시도 (qwe.py에서는 soup.find("h1") 사용, 여기서는 기존 선택자 유지)
            title_tag_chosun = soup.select_one('h1.article-header__headline') 
            if not title_tag_chosun: 
                title_tag_chosun = soup.select_one('div.article_title h1') 
            
            if title_tag_chosun and title == "제목을 찾을 수 없습니다.":
                for unwanted_span in title_tag_chosun.select('span.article-header__headline-badge_premium'): 
                    unwanted_span.decompose()
                title = title_tag_chosun.get_text(strip=True)
                print(f"Chosun parser (Post-Selenium): Title extracted from h1: {title[:50]}...")
            # 조선일보 제목은 og:title로 가져오는 경우가 많으므로, 여기서 못찾아도 일반 파서에서 찾을 수 있음.
            # 단, JS 렌더링 후의 HTML 구조에서 가져올 수 있다면 더 정확.

            # 본문 추출 시도 (qwe.py에서는 p.article-body__content-text 사용, 기존 선택자 활용)
            content_section_chosun = soup.select_one('section.article-body') 
            if not content_section_chosun: 
                content_section_chosun = soup.select_one('div.article_body, div.article_content') 
                if content_section_chosun:
                     print(f"Chosun parser (Post-Selenium): Found body with alternative selector: {content_section_chosun.name}.{content_section_chosun.get('class')}")
            
            if content_section_chosun and (title != "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip())): 
                print(f"Chosun parser (Post-Selenium): Found content section: {content_section_chosun.name}.{content_section_chosun.get('class')}")
                # 광고, 이미지 캡션, 관련 기사 등 제거 (기존 선택자 및 주석 유지)
                for unwanted_chosun in content_section_chosun.select(
                    'script, style, iframe, .flex-chain-wrapper, figure.article-body__content-image figcaption, .article-copyright, .article-tags, .separator-container, .raw-html, .social-share, .author-info, .author_info, .promotion, .related_news, .ad_box, .ad_wrapper, .dfnAd, div[class*="__ad-container"], div[class*="box_bottom_ad"], div[id*="google_ads"], div[data-widget_id], div.related_video_widget, div.news_bottom_notice, .text--grey, .ic_lock' # .text--grey는 기자 이름, .ic_lock은 프리미엄 아이콘
                ):
                    # print(f"Chosun parser: Removing unwanted element: {unwanted_chosun.name}.{unwanted_chosun.get('class')}") # 기존 상세 로그 유지 가능
                    unwanted_chosun.decompose()
                
                # 이미지 자체도 제거 (텍스트만 남기기 위함). (기존 주석 유지)
                for img_tag_chosun in content_section_chosun.select('figure.article-body__content-image img, .article_body_content-image img'):
                    parent_figure = img_tag_chosun.find_parent('figure')
                    if parent_figure:
                        # print(f"Chosun parser: Removing image within figure: {parent_figure.name}.{parent_figure.get('class')}") # 기존 상세 로그 유지 가능
                        parent_figure.decompose() 
                    else:
                        img_tag_chosun.decompose() 

                for br_tag in content_section_chosun.find_all("br"):
                    br_tag.replace_with('\n')
                
                # 우선적으로 .article-body__content-text 클래스를 가진 p 태그에서만 본문 추출 (qwe.py와 유사)
                paragraphs_chosun = content_section_chosun.select('p.article-body__content-text')
                
                if not paragraphs_chosun: 
                    print("Chosun parser (Post-Selenium): 'p.article-body__content-text' not found, trying fallback find_all('p') in article-body.") # 기존 로그 유지
                    paragraphs_chosun = content_section_chosun.find_all('p', recursive=False) # 바로 밑 p들만 (기존 주석 유지)

                content_texts_chosun = []
                for p_tag_chosun in paragraphs_chosun:
                    # 이미 제거된 figcaption 내부의 p태그나, 다른 불필요한 p태그를 걸러내기 위한 추가 조건 (기존 주석 유지)
                    if p_tag_chosun.find_parent(class_=[re.compile(r"ad-"), "article-copyright", "article-tags", "author-info", "figcaption"]):
                        continue
                    if p_tag_chosun.find('table') or p_tag_chosun.find('div', recursive=False):
                        continue
                    
                    p_text = p_tag_chosun.get_text(separator='\n', strip=True)
                    # 특정 불필요한 문구 포함 시 제외 (기존 주석 유지)
                    if p_text and (len(p_text) < 10 and (p_text.startswith('▲') or p_text.startswith('●') or p_text.startswith('■')) and p_text.count(p_text[0]) == len(p_text)):
                        # print(f"Chosun parser: Skipping short decorative line: {p_text}") # 기존 상세 로그 유지 가능
                        continue
                    if p_text:
                        content_texts_chosun.append(p_text)
                
                if content_texts_chosun and (content == "본문 내용을 찾을 수 없습니다." or not content.strip()): # 본문 비었을 때만
                    content = '\n'.join(content_texts_chosun).strip()
                    print(f"Chosun parser (Post-Selenium): Extracted content length: {len(content)}") # 기존 로그 유지
                elif not content_texts_chosun and (content == "본문 내용을 찾을 수 없습니다." or not content.strip()):
                    print("Chosun parser (Post-Selenium): No content extracted from <p> tags within article-body after filtering.") # 기존 로그 유지
            elif content_section_chosun and not (content == "본문 내용을 찾을 수 없습니다." or not content.strip()): 
                 print("Chosun parser (Post-Selenium): section.article-body found, but content already extracted or title not found.") # 기존 로그 유지 (조건 수정)
            elif not content_section_chosun and 'chosun.com' in url.lower(): # 조선일보 URL인데 섹션을 못찾은 경우
                print("Chosun parser (Post-Selenium): section.article-body (and alternatives) not found in Selenium-rendered HTML.") # 수정된 로그

        # 2. 연합뉴스 시도
        if title == "제목을 찾을 수 없습니다." or (content == "본문 내용을 찾을 수 없습니다." or not content.strip()):
            title_tag_yna = soup.find('h1', class_=['tit', 'title'])
            if not title_tag_yna: title_tag_yna = soup.find('h2', class_='title') # 일부 연합뉴스 기사 h2 사용
            if title_tag_yna and title == "제목을 찾을 수 없습니다.": title = title_tag_yna.get_text(strip=True)

        # 연합뉴스 본문 파싱은 제목 상태와 별개로, 본문이 비어있으면 시도
        if content == "본문 내용을 찾을 수 없습니다." or not content.strip():
            content_div_yna = soup.find('article', id='dic_area') # 연합뉴스 본문 영역 id
            if not content_div_yna: content_div_yna = soup.find('div', class_=['story-news', 'article-txt']) # 대체 선택자
            if content_div_yna : # 본문 영역 찾았으면 진행 (제목 조건은 위에서 이미 처리)
                for ad_selector in content_div_yna.select(".ad_wrap, .social-sns, script, style, .promotion_area, .related_news_area, .copyright, .reporter_area, figure.title_img, .view_setting_area, .banner_area, #scrollEnd"):
                    ad_selector.decompose()
                for br_tag in content_div_yna.find_all("br"):
                    br_tag.replace_with('\n') 
                
                content_texts_yna = []
                # 연합뉴스는 주로 p 태그로 본문 구성
                paragraphs_yna = content_div_yna.find_all('p', recursive=True)
                if paragraphs_yna:
                    for p_yna in paragraphs_yna:
                        p_text_yna = p_yna.get_text(separator='\n', strip=True)
                        if p_text_yna and not p_yna.find_parent(class_=["b_center", "b_caption", "ad_center_fixed"]) : # 캡션, 광고 등 제외
                             content_texts_yna.append(p_text_yna)
                
                if not content_texts_yna: # p태그에서 못찾았으면 전체 텍스트 시도
                    all_text_yna = content_div_yna.get_text(separator='\n', strip=True)
                    if all_text_yna.strip(): content_texts_yna = [all_text_yna]

                if content_texts_yna: content = '\n'.join(content_texts_yna) # 본문이 비었을때만 덮어쓰므로, 이전 content값 고려 필요 없음

        # 3. 일반적인 메타 태그 또는 body 전체에서 p 태그 시도 (최후의 수단)
        # 제목 폴백: 모든 사이트별 파서가 제목을 못 찾은 경우
        if title == "제목을 찾을 수 없습니다.": 
            meta_title_og = soup.find('meta', property='og:title')
            if meta_title_og and meta_title_og.get('content'):
                title = meta_title_og['content']
            else:
                meta_title_name = soup.find('meta', attrs={'name': 'title'})
                if meta_title_name and meta_title_name.get('content'):
                    title = meta_title_name['content']
                elif soup.title and soup.title.string:
                    title = soup.title.string.strip()

        # 본문 폴백: 모든 사이트별 파서가 본문을 못 찾았거나 내용이 없는 경우
        if content == "본문 내용을 찾을 수 없습니다." or not content.strip():
            print("All site-specific parsers failed for content, trying body p tags as a fallback...")
            body_tag = soup.find('body')
            if body_tag:
                # 불필요한 태그 제거 목록 확장 (.layer_wrap 등 추가)
                for unwanted in body_tag.select('script, style, nav, footer, header, aside, form, .banner, .ad, .advertisement, #comment, .comments, .sidebar, .viewTool, .viewHeader, .articleBottomNews, .articleBottomAD, .sideOrderList, .newsBrand, .vote, .fixedHeaderShow, #floatPopup, #bottomScrollBox, .go_top, .layer_wrap, .header_util, .news_nav, .footer_nav'):
                    unwanted.decompose()
                
                for br_tag in body_tag.find_all("br"): 
                    br_tag.replace_with('\n') 

                paragraphs = body_tag.find_all('p')
                # p 태그에서 텍스트 추출 시 부모 태그 조건 추가 ('button', 'li' 등)
                content_texts_fallback = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True) and len(p.get_text(strip=True)) > 30 and not p.find_parent(['figcaption', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'li'])] 
                
                if content_texts_fallback:
                    content = '\n'.join(content_texts_fallback) 
                else: # p 태그에서 못찾으면 body 전체 텍스트에서 의미있는 라인 추출
                    all_body_text = body_tag.get_text(separator='\n', strip=True) 
                    lines = [line.strip() for line in all_body_text.splitlines() if line.strip()]
                    # 의미있는 라인 필터링 조건 강화 (구독, 댓글, 구분선 등 제거)
                    meaningful_lines = [line for line in lines if len(line) > 15 and not (line.startswith("Copyright") or line.startswith("ⓒ") or "무단 전재" in line or "재배포 금지" in line or "구독하기" in line or "기사제공" in line or "댓글" in line or line.count("-") > 5 or line.count("|") > 3 )]
                    if meaningful_lines:
                        content = '\n'.join(meaningful_lines) 
        
        # 최종 정리: 모든 파싱 시도 후 공통으로 적용 (기존 로직 및 주석, 확장된 사이트 목록 유지)
        if title: title = ' '.join(title.split()) # 연속 공백 제거
        if content and content != "본문 내용을 찾을 수 없습니다.": 
            lines = [line.strip() for line in content.splitlines()] 
            meaningful_lines = []
            # 바이라인 및 불필요한 라인 제거를 위한 정규식 패턴 확장
            byline_patterns = [r"^[^\s]+\s*기자", r"^\s*\(?\s*(?:특파원|사진|영상|자료|글|취재|편집|정리)\s*=?\s*[^\s\)]+\s*(?:기자|특파원|위원|연구원|PD|작가|교수|센터장|객원기자|논설위원|주필|편집장|대표|사장|팀장|부장|차장|과장|국장|데일리안|연합뉴스|뉴스1|뉴시스|이데일리|서울신문|엑스포츠뉴스|조선일보)[^@]*$", r".*@.*\.(?:com|co\.kr|net|kr|go\.kr|ac\.kr|or\.kr)$", r"ⓒ\s*.*뉴스", r"Copyrights\s*ⓒ", r"무단 전재|재배포 금지|배포금지", r"기사제보\s*:", r"제보\s*:", r"독자여러분", r"^사진\s*=", r"^영상캡처=", r"<저작권자 © .* 무단전재 및 재배포금지>", r"^\s*▶\s*", r"^\s*◆\s*", r"^\s*저작권자\s*\(c\)", r"^[가-힣]{2,5}\s*기자\s*=", r"^\s*\[[가-힣]+\s*=\s*[가-힣]+\]", r"^\s*\([가-힣]+\s*=\s*[가-힣]+\)", r"^\s*[가-힣·]+\s+[가-힣]{2,4}\s*기자"]
            temp_final_lines = []

            for line_text in lines:
                if not line_text: continue # 빈 줄 무시
                 # 너무 짧고 의미 없는 라인 (예: 구분선 '--', 특수문자로만 구성) 제거 강화
                if len(line_text) < 10 and not re.search(r'[가-힣a-zA-Z0-9]', line_text): # 숫자나 글자 없이 10자 미만이면
                    if len(line_text) < 5 and not line_text[0].isalnum() and not line_text[-1].isalnum() and line_text.count(line_text[0]) == len(line_text): continue
                
                is_byline = False # 바이라인 패턴 검사
                for pattern in byline_patterns: 
                    if re.search(pattern, line_text, re.IGNORECASE): is_byline = True; break
                if is_byline and len(line_text) < 100: continue # 짧은 바이라인일 경우 제거
                
                temp_final_lines.append(line_text)
            
            if temp_final_lines: # 임시 라인들이 있을 경우, 끝에서부터 추가 제거 로직
                lines_to_check_at_end = min(5, len(temp_final_lines)); lines_to_remove_count = 0
                for i in range(1, lines_to_check_at_end + 1): # 마지막 N줄 검사
                    line_to_inspect = temp_final_lines[-i]; is_trailing_junk = False
                    for pattern in byline_patterns: # 바이라인 패턴 재확인
                        if re.search(pattern, line_to_inspect, re.IGNORECASE): is_trailing_junk = True; break
                    # 추가적인 꼬리 광고/안내 문구 패턴
                    if not is_trailing_junk and (line_to_inspect.lower().startswith(("onlinenews@", "online@", "digital@")) or "official@" in line_to_inspect.lower() or re.search(r"구독\\s*바로가기", line_to_inspect) or re.search(r"네이버에서 [가-힣]+를 구독하세요", line_to_inspect) or re.search(r"채널 추가하기", line_to_inspect) or (len(line_to_inspect) < 20 and ( "클릭" in line_to_inspect or "확인" in line_to_inspect))):
                        is_trailing_junk = True
                    
                    if is_trailing_junk and len(line_to_inspect) < 150: lines_to_remove_count = i # 제거할 라인 수 업데이트
                    elif lines_to_remove_count > 0 and not is_trailing_junk: lines_to_remove_count = i -1; break # 정크 아닌 라인 만나면 이전까지만 제거
                    elif not is_trailing_junk: break # 정크 아니면 검사 중단
                
                if lines_to_remove_count > 0: meaningful_lines = temp_final_lines[:-lines_to_remove_count]
                else: meaningful_lines = temp_final_lines
            else: # temp_final_lines 자체가 비었으면 meaningful_lines도 비게 됨
                 meaningful_lines = []

            if meaningful_lines: content = '\n'.join(meaningful_lines).strip() # 정리된 라인들을 단일 개행으로 합침
            else: content = "본문 내용을 찾을 수 없습니다." # 모든 라인이 제거된 경우
        elif content != "본문 내용을 찾을 수 없습니다." and not content.strip(): # 내용이 있지만 공백뿐인 경우
            content = "본문 내용을 찾을 수 없습니다."
        elif not content: # content가 None 등으로 비어있는 경우 (이론상 발생 안해야함)
             content = "본문 내용을 찾을 수 없습니다."

        
        # 제목에서 사이트 이름 등 제거 (기존 로직 및 주석, 확장된 사이트 목록 유지)
        if title and title != "제목을 찾을 수 없습니다.":
            original_title = title # 원본 제목 보존 (제거 후 너무 짧아지거나 이상해지면 복구용)
            common_separators = [" | ", " - ", " :: ", " < ", " > "] 
            for sep in common_separators:
                if sep in title: 
                    parts = title.split(sep)
                    # 가장 긴 부분을 제목으로 선택하되, 너무 짧으면 다른 부분을 고려
                    parts.sort(key=len, reverse=True)
                    if parts:
                        potential_title = parts[0].strip()
                        if len(potential_title) > 5: # 최소 길이 만족 시
                            title = potential_title
                        elif len(parts) > 1 and len(parts[1].strip()) > 5 : # 두번째로 긴 부분이 조건 만족 시
                            title = parts[1].strip()
                        # 위에서 break 하지 않고, 다른 구분자로 더 짧게 만들 수 있는지 계속 확인하는 로직은 아님.
                        # 하나의 구분자로 분리 후 가장 적절한 부분을 선택하고, 다른 구분자는 더 이상 고려 안 함.
                        # 만약, 여러 구분자 고려 후 최적을 찾아야 한다면 로직 변경 필요. 현재는 첫 발견 구분자로 처리.
                        break # 첫 번째 유효한 구분자로 처리 후 종료
            
            # 사이트 이름 목록 확장 및 정렬 (긴 이름부터 제거 시도)
            site_names_to_remove = ["엑스포츠뉴스", "서울신문", "조선일보", "연합뉴스", "연합뉴스TV", "이데일리", "YTN", "KBS", "MBC", "SBS", "JTBC", "MBN", "TV조선", "채널A", "뉴스1", "뉴시스", "머니투데이", "한국경제", "매일경제", "헤럴드경제", "아시아경제", "파이낸셜뉴스", "전자신문", "ZDNet Korea", "ZDNet", "지디넷코리아", "블로터", "디지털데일리", "아이뉴스24", "미디어오늘", "오마이뉴스", "프레시안", "한겨레", "경향신문", "국민일보", "동아일보", "세계일보", "중앙일보", "한국일보", "노컷뉴스", "데일리안", "더팩트", "위키트리", "인사이트", "허프포스트코리아", "민중의소리", "데일리NK", "통일뉴스", "코리아헤럴드", "코리아중앙데일리", "코리아타임스"]
            site_names_to_remove = list(set(site_names_to_remove)) # 중복 제거
            site_names_to_remove.sort(key=len, reverse=True) # 긴 이름부터 처리

            for site_name in site_names_to_remove:
                # 제목 끝 부분 일치
                if title.endswith(f" {site_name}") or title.endswith(f"-{site_name}") or title.endswith(f"|{site_name}") or title.endswith(f":{site_name}") or title.endswith(site_name):
                    title = title.rsplit(site_name, 1)[0].strip()
                    if title.endswith(("|", "-", ":", "<", ">")): title = title[:-1].strip() # 남은 구분자 정리
                # 제목 시작 부분 일치
                if title.startswith(f"{site_name} ") or title.startswith(f"{site_name}-") or title.startswith(f"{site_name}|") or title.startswith(f"{site_name}:") or title.startswith(site_name):
                    title = title.split(site_name, 1)[-1].strip()
                    if title.startswith(("|", "-", ":", "<", ">")): title = title[1:].strip() # 남은 구분자 정리
                # 괄호 안의 사이트 이름 제거
                title = title.replace(f"({site_name})", "").replace(f"[{site_name}]", "").replace(f"<{site_name}>", "").strip()
                title = ' '.join(title.split()) # 사이트명 제거 후 발생할 수 있는 연속 공백 제거

            if not title.strip() or len(title.strip()) < 5 : # 너무 짧아지거나 비었으면 원본 제목으로 복구
                title = original_title
        return title, content

    except requests.exceptions.RequestException as e: # requests 관련 예외 처리
        print(f"URL 요청 오류 (requests): {url}, {e}") 
        return "오류", f"기사를 가져오는 중 오류가 발생했습니다 (requests): {e}"
    except Exception as e: # Selenium 오류 및 기타 모든 예외 처리
        print(f"기사 처리 중 예외 발생: {url}, {e}") 
        import traceback
        traceback.print_exc() 
        return "오류", f"기사 내용을 처리하는 중 오류가 발생했습니다: {e}" 
    finally:
        # if session: # HTMLSession 사용 시 세션 닫기 -> 제거됨
        #     session.close()
        #     print(f"HTMLSession closed for {url}")
        if driver: # Selenium WebDriver 사용 시 드라이버 종료
            try:
                driver.quit()
                print(f"Selenium WebDriver closed for {url}")
            except Exception as e_quit:
                print(f"Error closing Selenium WebDriver for {url}: {e_quit}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    url = request.form.get('news_url')
    if not url:
        return render_template('result.html', error="URL을 입력해주세요.")

    print(f"Processing URL: {url}")
    title, content = fetch_article_content(url)
    # print(f"Title from fetch: {title}") # fetch_article_content 이후 title 로깅은 아래에서 하므로 중복
    error_message = None
    title_status = title # Preserve original title status for messages (오류, 제목없음 등)

    if title == "오류": # Critical error from fetch_article_content itself
        error_message = content # content variable holds the actual error string from fetch
    # fetch_article_content에서 반환된 content가 특정 오류 메시지 패턴으로 시작하는지 확인
    elif content.startswith("기사를 가져오는 중 오류가 발생했습니다") or content.startswith("기사 내용을 처리하는 중 오류가 발생했습니다"):
        error_message = content # Specific error message passed up from fetch_article_content
    else:
        # Check for extraction failures if no critical error occurred
        title_extraction_failed = (title == "제목을 찾을 수 없습니다.")
        content_extraction_failed = (content == "본문 내용을 찾을 수 없습니다." or not content.strip())

        if title_extraction_failed and content_extraction_failed:
            error_message = "기사 제목과 내용을 모두 추출할 수 없었습니다. 해당 URL의 기사 구조를 지원하지 않거나, 일시적인 네트워크 문제일 수 있습니다."
        elif content_extraction_failed:
            # Content failed, but title might be okay.
            display_title_in_error = title_status if not title_extraction_failed else "(제목 없음)"
            error_message = f"추출된 기사 내용이 없습니다. (제목: '{display_title_in_error}'). 콘솔(터미널) 로그에서 'fetch_article_content' 함수의 상세 내용을 확인해주세요."
        elif title_extraction_failed:
            # Title failed, but content is okay. This is not necessarily a fatal error for summarization.
            # We will let it proceed, and the result page will show "제목을 찾을 수 없습니다."
            # If title is absolutely required later (e.g., for clickbait model strict mode), this might change.
            pass # No error_message here, proceed to try summarization

    if error_message:
        # Determine display_title for error page based on original title_status
        display_title_on_error_page = title_status if title_status not in ["오류", "제목을 찾을 수 없습니다."] else "제목 추출 실패"
        return render_template('result.html', error=error_message, url=url, title=display_title_on_error_page, summary="", prediction="판단 불가")
    
    try:
        summary = summarize_text(content) # 여기서 content는 비어있지 않음 (위에서 content_extraction_failed 시 error_message 처리됨)
        print(f"Title for summarization/prediction: {title}") # 여기서 title, summary 로깅
        print(f"content: {content}")
        print(f"Raw Content Length for Summary: {len(content)}") # 요약 전 원문 길이 로깅 추가
        print(f"Summary: {summary}")
    
        # 요약된 내용을 낚시성 판단 모델에 사용 (이전 대화에서 원본 내용 사용으로 변경되었던 부분 확인 필요 -> 현재는 요약본 사용으로 유지)
        clickbait_prediction_result = predict_clickbait(title, summary) 
        print(f"Clickbait Prediction: {clickbait_prediction_result}")

        return render_template('result.html', url=url, title=title, summary=summary, prediction=clickbait_prediction_result)
    except Exception as e:
        print(f"An error occurred during summarization/prediction: {e}") # 오류 메시지 구체화
        user_error_message = f"기사 처리 중 오류가 발생했습니다. (세부 정보: {str(e)})"
        import traceback
        traceback.print_exc() 
        return render_template('result.html', error=user_error_message, url=url, title=title, summary="오류로 인해 요약 불가", prediction="판단 불가")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 