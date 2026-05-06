import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class BERTDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1', use_fast=False)
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
)
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def data_load_and_preprocess(data_root_path, train_file_name, valid_file_name, test_file_name=None):
    # ======= 1. 데이터 로드 =======
    train_df = pd.read_csv(os.path.join(data_root_path, train_file_name), encoding="utf-8")
    valid_df = pd.read_csv(os.path.join(data_root_path, valid_file_name), encoding="utf-8")
    if test_file_name is not None:
        test_df = pd.read_csv(os.path.join(data_root_path, test_file_name), encoding="utf-8")

    X_train_text = train_df["text"]
    y_train_text = train_df["label"]

    X_valid_text = valid_df["text"]
    y_valid_text = valid_df["label"]

    if test_df is not None:
        X_test_text = test_df["text"].astype(str)
        y_test_text = test_df["label"].astype(str).str.strip()

    # ======= 2. 레이블 인코딩 =======
    custom_class_order = ['안전기준 미준수', '운송차량', '시설 결함']
    mapping = {
        '안전기준 미준수': 'C₁',
        '운송차량': 'C₂',
        '시설 결함': 'C₃'
    }

    label_encoder = LabelEncoder()
    label_encoder.fit(custom_class_order)
    y_train = label_encoder.transform(y_train_text)
    y_valid = label_encoder.transform(y_valid_text)
    if test_df is not None:
        y_test = label_encoder.transform(y_test_text)

    class_names = [mapping[c] for c in custom_class_order]

    # ======= 3. 불용어 로드 =======
    path = os.path.join(data_root_path, 'korean_stopwords.txt')
    with open(path, 'r', encoding='utf-8') as f:
        korean_stopwords = set(line.strip() for line in f)

    # ======= 4. 텍스트 전처리 =======
    from konlpy.tag import Okt
    okt = Okt()

    def preprocess(text):
        text = re.sub(r'\b\d+\b', '', text)
        tokens = okt.morphs(text, stem=True)
        tokens = [t for t in tokens if t not in korean_stopwords]
        return " ".join(tokens)

    X_train_cleaned = X_train_text.apply(preprocess)
    X_valid_cleaned = X_valid_text.apply(preprocess)

    if test_df is None:
        return X_train_cleaned, y_train, X_valid_cleaned, y_valid, class_names

    X_test_cleaned = X_test_text.apply(preprocess)
    return X_train_cleaned, y_train, X_valid_cleaned, y_valid, X_test_cleaned, y_test, class_names

def prepare_dataset_kobert(data_root_path, train_file_name, valid_file_name, test_file_name=None, seed=None, batch_size=16):
    if test_file_name is None:
        X_train_cleaned, y_train, X_valid_cleaned, y_valid, class_names = data_load_and_preprocess(
            data_root_path, train_file_name, valid_file_name
        )
    else:
        X_train_cleaned, y_train, X_valid_cleaned, y_valid, X_test_cleaned, y_test, class_names = data_load_and_preprocess(
            data_root_path, train_file_name, valid_file_name, test_file_name
        )
    
    train_dataset = BERTDataset(X_train_cleaned.tolist(), y_train)
    valid_dataset  = BERTDataset(X_valid_cleaned.tolist(), y_valid)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    valid_loader  = DataLoader(valid_dataset, batch_size=batch_size, generator=generator)

    if test_file_name is None:
        return train_dataset, valid_dataset, train_loader, valid_loader, class_names

    test_dataset = BERTDataset(X_test_cleaned.tolist(), y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader, class_names

def prepare_dataset_classicML(data_root_path, train_file_name, valid_file_name, test_file_name=None):
    if test_file_name is None:
        X_train_cleaned, y_train, X_valid_cleaned, y_valid, class_names = data_load_and_preprocess(
            data_root_path, train_file_name, valid_file_name
        )
    else:
        X_train_cleaned, y_train, X_valid_cleaned, y_valid, X_test_cleaned, y_test, class_names = data_load_and_preprocess(
            data_root_path, train_file_name, valid_file_name, test_file_name
        )

    vectorizer = TfidfVectorizer(max_features=1000, token_pattern=r'(?u)\b[^\d\W]+\b')
    X_train_vec = vectorizer.fit_transform(X_train_cleaned)
    X_valid_vec   = vectorizer.transform(X_valid_cleaned)

    if test_file_name is None:
        return X_train_vec, y_train, X_valid_vec, y_valid, class_names, vectorizer

    X_test_vec = vectorizer.transform(X_test_cleaned)
    return X_train_vec, y_train, X_valid_vec, y_valid, X_test_vec, y_test, class_names, vectorizer