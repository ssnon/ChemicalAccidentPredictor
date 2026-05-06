import torch.nn as nn
from kobert_transformers import get_kobert_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ======= 6. KoBERT 모델 정의 =======
class KoBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = get_kobert_model()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1]  # CLS pooled output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

def prepare_model(model_name, device, seed=42):
    if model_name == 'kobert':
        model = KoBERTClassifier(num_classes=3).to(device)
    elif model_name == 'LR':
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == 'RF':
        model = RandomForestClassifier(random_state=seed)
    elif model_name == 'SVM':
        model = SVC(probability=True, random_state=seed)
    elif model_name == 'XGBoost':
        model = XGBClassifier(eval_metric='mlogloss', random_state=seed, use_label_encoder=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model