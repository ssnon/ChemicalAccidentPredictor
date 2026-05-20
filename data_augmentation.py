import pandas as pd
import random
import asyncio
import time
import nest_asyncio
from googletrans import Translator
import numpy as np
from difflib import SequenceMatcher
import argparse
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='data augmentation through back translation')
parser.add_argument('--data_directory', default="dataset/", type=str,
                    help='data directory')
parser.add_argument('--input_file_name', default="train/train_raw.csv", type=str,
                    help='augmentation target')
parser.add_argument('--output_file_name', default="train/train.csv", type=str,
                    help='augmentation result')
parser.add_argument('--aug_ratio', default=1.3, type=float,
                    help='augmentation ratio')
parser.add_argument('--semantic_sim_threshold_lowerbound', default=0.90, type=float,
                    help='augmentation acceptance threshold')
parser.add_argument('--surface_sim_threshold_upperbound', default=0.96, type=float,
                    help='augmentation acceptance threshold')
parser.add_argument('--surface_sim_threshold_lowerbound', default=0.70, type=float,
                    help='augmentation acceptance threshold')

args = parser.parse_args()
# CSV 파일 로드
csv_path = os.path.join(args.data_directory, args.input_file_name)
df = pd.read_csv(csv_path)

# Google 번역기 초기화
translator = Translator()

# Back Translation 함수 (비동기)
async def back_translate_google_async(text, pivot_lang):
    ko_to_piv = await translator.translate(text, src='ko', dest=pivot_lang)
    piv_to_ko = await translator.translate(ko_to_piv.text, src=pivot_lang, dest='ko')
    return piv_to_ko.text

#def text_similarity(a, b):
#    return SequenceMatcher(None, a, b).ratio()

def compact_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    # 한글, 영어, 숫자만 남김
    s = re.sub(r"[^0-9a-zA-Z가-힣]+", "", s)
    return s

NUM_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*"
    r"(?:"
    r"ton|톤|kg|㎏|g|mg|"
    r"ml|mL|ML|l|L|ℓ|리터|"
    r"m3|m\^3|m³|㎥|"
    r"m2|m\^2|m²|㎡|"
    r"도|℃|°C|"
    r"%|"
    r"명|개|기|포|대|"
    r"시간|분"
    r")?",
    re.IGNORECASE
)

def normalize_num_unit(x: str) -> str:
    x = x.lower()
    x = re.sub(r"\s+", "", x)

    x = x.replace("톤", "ton")
    x = x.replace("㎏", "kg")
    x = x.replace("리터", "l")
    x = x.replace("ℓ", "l")

    x = x.replace("m^3", "m3")
    x = x.replace("m³", "m3")
    x = x.replace("㎥", "m3")

    x = x.replace("m^2", "m2")
    x = x.replace("m²", "m2")
    x = x.replace("㎡", "m2")

    x = x.replace("°c", "℃")

    return x

def extract_num_units(text: str):
    items = NUM_UNIT_RE.findall(str(text))
    return sorted(normalize_num_unit(x) for x in items)

def contains_any(text: str, words):
    return any(w in str(text) for w in words)

# original에 있는 주요 단어가 augmented에 얕게 바뀐게 아니면 reject (예: 누출 -> leak는 허용. 이외의 심한 변화는 reject)
def keyword_group_check(original: str, augmented: str, keyword_groups):
    for group in keyword_groups:
        if contains_any(original, group):
            if not contains_any(augmented, group):
                return False, group

    return True, None

KEYWORD_GROUPS = [
    ["누출", "Leak", "leak", "새는", "유출"],
    ["노출", "접촉"],
    ["방출", "배출"],
    ["파손", "손상", "파열"],
    ["폭발"],
    ["화재", "발화"],
    ["질식"],
    ["중독"],
    ["MDI합성공정"],
    ["PTXV825"],
    ["spinner"],
    ["신너"],
    ["안구"],
    ["황산"],
    ["염산"],
    ["질산"],
    ["생물 표본병"],
    ["표본액"],
    ["에스케이케미컬"],
    ["양극재"],
    ["산화코발트리튬망간니켈"],
    ["삼원"],
    ["마산"],
    ["창원시"],
    ["부상", "상해", "화상", "경상", "사상", "인명피해"],
]

sim_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
def semantic_cosine_similarity(a: str, b: str) -> float:
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)

    embeddings = sim_model.encode(
        [a, b],
        normalize_embeddings=True
    )

    return float(np.dot(embeddings[0], embeddings[1]))

BAD_REPLACEMENTS = [
    ("수소반응기", "수소원자로"),
    ("반응기", "원자로"),
    ("이상반응", "부작용"),
    ("단속 의뢰", "간헐 의뢰"),
    ("교육관", "교육원"),
    ("오색정수장", "우세 정수장"),
    ("신너", "희석제"),
    ("안구", "눈에 주입"),
    ("생물 표본병", "생체검체병"),
    ("표본액", "검체액"),
    ("에스케이케미컬", "에스케케미컬"),
    ("양극재", "정극재"),
    ("MDI합성공정", "Drain 합성 공정"),
    ("과다 밸브조작", "과밸브 조작"),    
    ("배송기룸", "납품실"),
    ("신설", "신품"),
    ("취합작업", "수거 작업"),
    ("소방용수", "방화수"),
    ("집수조", "집수정"),
    ("폐액탱크", "폐기물탱크"),
    ("볼트 탈락", "볼트를 풀어"),
    ("볼트 탈락", "볼트를 풀어"),
    ("묽힘열", "희열"),
    ("우수로", "빗물"),
    ("병 및 플라스크", "질병이나 플라스크"),
    ("경상", "사상자"),
    ("부상", "사상"),
    ("적재물", "부하"),
    ("용기", "컨테이너"),
    ("대보실업", "대보산업"),
]

BAD_PHRASES = [
    "따냈다고",
    "방전 선반",
    "300L 정보",
    "방전벽",
    "배출벽",
    "퇴적까지",
    "발생하였습니다. 300L",
    "눈에 주입",
    "spinner).",
    "질산이 검출",
    "폐기되었습니다",
    "Drain 합성 공정",
    "발생현황",
    "생체검체",
    "에스케케미컬",
    "과밸브",
    "B. 화재",
    "작업자가 반드시",
    "납품실",
    "원료혼합시 누수",
    "옛부터",
    "소방추산치",
    "상업용 현장",
    "신품 부생복합발전",
    "황가스"
    "희열에 의해",
    "끓여 작업자",
    "질병이나 플라스크",
    "사상자 7명 및 경상",
    "6명 사상 및 경상",
    "정도가 발생하였습니다",
    ". 트리클로로실란.",
    "누출사고는",
    "부하(약",
    "산성 탱크.",
    "No까지",
    "폭발 필터",
    "탱크로로",
]

def bad_replacement_check(original: str, augmented: str):
    # 특정 위험 치환 검사
    for src, bad in BAD_REPLACEMENTS:
        if src in original and bad in augmented:
            return False, f"bad replacement: {src} -> {bad}"

    # 증강문 자체의 이상 표현 검사
    for bad_phrase in BAD_PHRASES:
        if bad_phrase in augmented:
            return False, f"bad phrase: {bad_phrase}"

    return True, None

CRITICAL_TERMS = [
    "적린",
    "염소산칼륨",
    "방류턱",
    "방류벽",
    "방류조",
    "방유제",
    "증착공정",
    "수소반응기",
    "염화수소",
    "과산화수소",
    "수산화나트륨",
    "수산화칼륨",
    "암모니아",
]

# 변형되면 안되는 주요 단어 체크
def critical_term_check(original: str, augmented: str):
    for term in CRITICAL_TERMS:
        if term in original and term not in augmented:
            return False, f"critical term missing: {term}"
    return True, None

def IntegrityTest_for_augmentedData(original_text, augemented_text):
    original = "" if original_text is None else str(original_text)
    augmented = "" if augemented_text is None else str(augemented_text)
    #sim = cosine_similarity(original, augmented)
    semantic_sim = semantic_cosine_similarity(original, augmented)
    surface_sim = SequenceMatcher(None, compact_text(original), compact_text(augmented)).ratio()

    # 빈 문장 방지
    if not augmented.strip():
        return False, "empty augmented text"

    # 공백/특수문자 제거 후 같으면 동일문장 reject
    if compact_text(original) == compact_text(augmented):
        return False, "almost identical after normalization"

    # 문장 유사도가 너무 낮으면 (즉, 너무 많이 바꼈으면) reject.
    if semantic_sim < args.semantic_sim_threshold_lowerbound:
        return False, f"similarity too low: {semantic_sim}"

    # 문장 표면상의 유사도가 너무 높으면 (띄어쓰기 하나 바꾸기, 조사 하나 바꾸기 등), 혹은 지나치게 낮으면 reject
    if surface_sim > args.surface_sim_threshold_upperbound or surface_sim < args.surface_sim_threshold_lowerbound:
        return False, f"similarity too high: {surface_sim}"

    # 문장 길이 크게 달라지면 reject
    o_len = len(original)
    a_len = len(augmented)
    if o_len == 0:
        return False, "original empty"

    ratio = a_len / o_len
    if ratio < 0.75 or ratio > 1.35:
        return False, f"length ratio abnormal: {ratio:.3f}"

    # 수치 및 단위 보존
    orig_nums = extract_num_units(original)
    aug_nums = extract_num_units(augmented)
    if orig_nums != aug_nums:
        return False, f"num/unit mismatch: {orig_nums} vs {aug_nums}"

    # 주요 단어들 보존
    ok, missing_group = keyword_group_check(original, augmented, KEYWORD_GROUPS)
    if not ok:
        return False, f"missing keyword group: {missing_group}"

    ok, missing_group = bad_replacement_check(original, augmented)
    if not ok:
        return False, f"missing keyword group: {missing_group}"
    
    ok, missing_group = critical_term_check(original, augmented)
    if not ok:
        return False, f"missing keyword group: {missing_group}"

    return True, "ok"

SPECIAL_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_\-\*/]*|"
    r"\d+(?:\.\d+)?\s*(?:ton|kg|g|mg|ml|mL|L|l|%)",
    re.IGNORECASE
)

def extract_special_tokens(text: str):
    return SPECIAL_TOKEN_RE.findall(str(text))

def protect_terms(text: str, terms):
    mapping = {}
    protected = str(text)

    # 긴 용어부터 먼저 치환해야 함
    terms = sorted(set(terms), key=len, reverse=True)

    for i, term in enumerate(terms):
        if term in protected:
            placeholder = f"ZXQTERM{i:04d}ZXQ"
            protected = protected.replace(term, placeholder)
            mapping[placeholder] = term

    return protected, mapping


def restore_terms(text: str, mapping):
    restored = str(text)

    for placeholder, term in mapping.items():
        restored = restored.replace(placeholder, term)

    return restored

async def main():
    #target_ratio = 1.3
    #similarity_threshold = 0.9
    # C1(시설 결함)에 해당하는 텍스트만 추출 / C2(안전기준 미준수)
    df_transport = df[df['label'] == '시설 결함']
    texts = df_transport['text'].tolist()  # 실제 텍스트 컬럼명 사용
    
    print(f"원본 문장 수: {len(texts)}")
  
    needed = int(len(texts) * args.aug_ratio - len(texts))
    print(f"필요한 증강 수: {needed}")

    pivot_langs = ["en", "ja", "zh-CN", "fr", "de", "es", "ru"]
    augmented_rows = []
    generated = 0
    candidates = texts.copy()

    random.shuffle(candidates)
    while(generated<needed):
        count = 0
        
        for original in candidates:
            accepted_bt = ""
            success = False

            count = count + 1
            if generated >= needed:
                break
            
            print(f"{len(candidates)} 중 {count}번째 문장")
            try:
                for pivot in pivot_langs:
                    auto_terms = extract_special_tokens(original)
                    terms = auto_terms
                    protected_original, mapping = protect_terms(original, terms)

                    bt = await back_translate_google_async(protected_original, pivot)
                    await asyncio.sleep(1)

                    bt = restore_terms(bt, mapping)
                    result, reason = IntegrityTest_for_augmentedData(original, bt)
                    if not result:
                        print(f"rejected : {reason}")
                        print(f"try agian...")
                        continue

                    accepted_bt = bt
                    accepted_reason = reason
                    success = True
                    break

                if not success:
                    print(f"[SKIP] 모든 pivot 실패: {original}")
                    continue

                print(f"원본 : {original}")
                print(f"변환 : {bt}")
                augmented_rows.append({
                    'text': accepted_bt,
                    'label': '시설 결함',
                    'original_text': original
                })

                generated += 1
                print(f"{generated} / {needed} 문장 증강 완료")

            except Exception as e:
                print(f"[BT 오류] {e}, 원문: {original}")
                continue
        print("reshuffle...")

    df_transport = df[df['label'] == '안전기준 미준수']
    texts = df_transport['text'].tolist()  # 실제 텍스트 컬럼명 사용

    print(f"원본 문장 수: {len(texts)}")
    needed = int(len(texts) * args.aug_ratio - len(texts))
    print(f"필요한 증강 수: {needed}")

    generated = 0
    candidates = texts.copy()
    random.shuffle(candidates)
    while(generated<needed):
        count = 0
        
        for original in candidates:
            accepted_bt = ""
            success = False
            count = count + 1
            if generated >= needed:
                break
            
            print(f"{len(candidates)} 중 {count}번째 문장")
            try:
                for pivot in pivot_langs:
                    auto_terms = extract_special_tokens(original)
                    terms = auto_terms
                    protected_original, mapping = protect_terms(original, terms)

                    bt = await back_translate_google_async(protected_original, pivot)
                    await asyncio.sleep(1)

                    bt = restore_terms(bt, mapping)
                    result, reason = IntegrityTest_for_augmentedData(original, bt)
                    if not result:
                        print(f"rejected : {reason}")
                        print(f"try agian...")
                        continue

                    accepted_bt = bt
                    accepted_reason = reason
                    success = True
                    break

                if not success:
                    print(f"[SKIP] 모든 pivot 실패: {original}")
                    continue

                print(f"원본 : {original}")
                print(f"변환 : {bt}")
                augmented_rows.append({
                    'text': accepted_bt,
                    'label': '안전기준 미준수',
                    'original_text': original
                })

                generated += 1
                print(f"{generated} / {needed} 문장 증강 완료")

            except Exception as e:
                print(f"[BT 오류] {e}, 원문: {original}")
                continue
        print("reshuffle...")
        

    print(f"최종 증강 문장 수: {len(augmented_rows)}")

    df_aug = pd.DataFrame(augmented_rows)
    # 원본 train.csv와 같은 컬럼만 사용
    df_combined = pd.concat(
        [df[['text', 'label']], df_aug[['text', 'label']]],
        ignore_index=True
    )

    df_combined.to_csv(
        os.path.join(args.data_directory, args.output_file_name),
        index=False,
        encoding='utf-8-sig'
    )
    for i, row in enumerate(augmented_rows[:10]):
        print(f"{i+1}")
        print(f"  원본   : {row['original_text']}")
        print(f"  증강문 : {row['text']}\n")


if __name__ == "__main__":
    asyncio.run(main())