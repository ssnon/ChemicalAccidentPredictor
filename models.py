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

def prepare_model(model_name, device, seed=42, args=None):
    if model_name == 'kobert':
        model = KoBERTClassifier(num_classes=3).to(device)
    elif model_name == 'LR':
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == 'RF':
        if args is None:
            model = RandomForestClassifier(random_state=seed)
        else:
            class_weight = args.rf_class_weight
            if class_weight == 'none':
                class_weight = None

            model = RandomForestClassifier(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_split=args.rf_min_samples_split,
                min_samples_leaf=args.rf_min_samples_leaf,
                max_features=args.rf_max_features,
                class_weight=class_weight,
                random_state=seed,
                n_jobs=-1
            )
        
    elif model_name == 'SVM':
        model = SVC(probability=True, random_state=seed)
    elif model_name == 'XGBoost':
        if args is None:
            model = XGBClassifier(eval_metric='mlogloss',random_state=seed,use_label_encoder=False)
        else:
            model = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                n_estimators=args.xgb_n_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                min_child_weight=args.xgb_min_child_weight,
                gamma=args.xgb_gamma,
                reg_lambda=args.xgb_reg_lambda,
                reg_alpha=args.xgb_reg_alpha,
                random_state=seed,
                n_jobs=-1,
                tree_method='hist',
                use_label_encoder=False
            )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model