import pandas as pd, numpy as np, re
from collections import Counter
from mecab import MeCab

def normalize(text):
    text = re.sub(r'[^가-힣A-Za-z0-9 ]', ' ', text)
    return text.lower()

tagger = MeCab()
df = pd.read_csv('test.csv', header=None)            # title, label
df.columns = ['label', 'title']
print(df)

# 1) 토큰화·정규화
df['tokens'] = df['title'].apply(
    lambda t: [w for w, pos in tagger.pos(normalize(t))
               if not pos.startswith(('J', 'E'))]
)

# 2) 빈도 카운트
cnt_c = Counter()
cnt_n = Counter()
for toks, y in zip(df['tokens'], df['label']):
    if y == 0:
        cnt = cnt_c
    else:
        cnt = cnt_n
    cnt.update(set(toks))   # 중복 단어는 1회만

# 3) 정보이득 계산
words = set(cnt_c) | set(cnt_n)
stats = []
for w in words:
    c, n = cnt_c[w], cnt_n[w]
    if c + n < 20:               # 희소 단어 제거
        continue
    p = c / (c+n)
    I = abs(p - 0.5)
    stats.append((w, c, n, p, I))
info = pd.DataFrame(stats, columns=['word','C','N','P','I'])
strong = info[info['I'] >= 0.2].sort_values('I', ascending=False)
strong.to_csv('strong_label_words.csv', index=False)
