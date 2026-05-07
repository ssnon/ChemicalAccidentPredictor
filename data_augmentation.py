import pandas as pd
import random
import asyncio
import time
import nest_asyncio
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher

# Jupyter 환경 이벤트 루프 문제 해결
nest_asyncio.apply()

# CSV 파일 로드
csv_path = '/home/hpc-ssu/PythonProjects/mimic_preprocessing/hajin/ChemicalAccident-test/kiwook/python/dataset/train/train.csv'
df = pd.read_csv(csv_path)

# Google 번역기 초기화
translator = Translator()

# Back Translation 함수 (비동기)
async def back_translate_google_async(text):
    en = await translator.translate(text, src='ko', dest='en')
    ko = await translator.translate(en.text, src='en', dest='ko')
    return ko.text

def text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


async def main():
    target_ratio = 1.3
    similarity_threshold = 0.9
    # C1(시설 결함)에 해당하는 텍스트만 추출 / C2(안전기준 미준수)
    df_transport = df[df['label'] == '시설 결함']
    texts = df_transport['text'].tolist()  # 실제 텍스트 컬럼명 사용
    
    print(f"원본 문장 수: {len(texts)}")
  
    needed = int(len(texts) * target_ratio - len(texts))
    print(f"필요한 증강 수: {needed}")

    augmented_rows = []
    generated = 0

    while generated < needed:
        original = random.choice(texts)

        try:
            bt = await back_translate_google_async(original)
            await asyncio.sleep(1)

            sim = text_similarity(original, bt)

            # 너무 달라진 문장은 제외
            if sim < similarity_threshold:
                continue

            augmented_rows.append({
                'text': bt,
                'label': '시설 결함',
                'original_text': original
            })

            generated += 1

            if generated % 10 == 0:
                print(f"{generated} / {needed} 문장 증강 완료")

        except Exception as e:
            print(f"[BT 오류] {e}, 원문: {original}")
            continue
    
    df_transport = df[df['label'] == '안전기준 미준수']
    texts = df_transport['text'].tolist()  # 실제 텍스트 컬럼명 사용

    print(f"원본 문장 수: {len(texts)}")
    needed = int(len(texts) * target_ratio - len(texts))
    print(f"필요한 증강 수: {needed}")

    generated = 0

    while generated < needed:
        original = random.choice(texts)

        try:
            bt = await back_translate_google_async(original)
            await asyncio.sleep(1)

            sim = text_similarity(original, bt)

            # 너무 달라진 문장은 제외
            if sim < similarity_threshold:
                continue

            augmented_rows.append({
                'text': bt,
                'label': '안전기준 미준수',
                'original_text': original
            })

            generated += 1

            if generated % 10 == 0:
                print(f"{generated} / {needed} 문장 증강 완료")

        except Exception as e:
            print(f"[BT 오류] {e}, 원문: {original}")
            continue
        

    print(f"최종 증강 문장 수: {len(augmented_rows)}")

    df_aug = pd.DataFrame(augmented_rows)
    df_aug.to_csv('/home/hpc-ssu/PythonProjects/mimic_preprocessing/hajin/ChemicalAccident-test/kiwook/python/dataset/train/train_augmented.csv', index=False, encoding='utf-8-sig')

    for i, row in enumerate(augmented_rows[:10]):
        print(f"{i+1}")
        print(f"  원본   : {row['original_text']}")
        print(f"  증강문 : {row['text']}\n")


if __name__ == "__main__":
    asyncio.run(main())