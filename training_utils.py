import os
import torch
import torch.optim as optim
from utils import progress_bar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import joblib
import json

class Trainer_pytorch():
    def __init__(self, early_stopping_patience=3, early_stopping_min_delta=0.0):
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.valid_loss_history = []
        self.valid_accuracy_history = []
        self.optimizer = None
        self.scheduler = None
        self.best_acc = 0

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0

    def train(self, epoch, model, train_loader, loss_function, device):
            print('\nEpoch: %d' % epoch)
            model.train()
            train_loss = 0
            correct = 0
            total = 0
                
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask)

                loss = loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                train_accuracy = 100.*correct/total

                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))        
            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_accuracy)
            
    def valid(self, epoch, model, valid_loader, loss_function, device, model_directory, model_name):
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = loss_function(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                valid_accuracy = 100.*correct/total

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))
            self.valid_loss_history.append(valid_loss)
            self.valid_accuracy_history.append(valid_accuracy)

        # Save checkpoint.
        if valid_accuracy > self.best_acc + self.early_stopping_min_delta:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'valid_accuracy': valid_accuracy,
                'epoch': epoch,
            }
            os.makedirs(os.path.join(model_directory, model_name, "checkpoint"), exist_ok=True)
            torch.save(state, os.path.join(model_directory, model_name, './checkpoint/ckpt.pth'))
            self.best_acc = valid_accuracy
            self.early_stopping_counter = 0
            print(f'Saving best checkpoint.. best_accuracy: {valid_accuracy:.6f}')    

            torch.save({
            'train_loss_history': self.train_loss_history,
            'train_accuracy_history': self.train_accuracy_history,
            'valid_loss_history': self.valid_loss_history,
            'valid_accuracy_history': self.valid_accuracy_history,
            }, os.path.join(model_directory,'history.pth')) 

        else:
            self.early_stopping_counter += 1
            print(
                f'EarlyStopping counter: {self.early_stopping_counter}/'
                f'{self.early_stopping_patience} | best_valid_accuracy: {self.best_acc:.6f}')

        should_stop = self.early_stopping_counter >= self.early_stopping_patience
        return valid_loss, valid_accuracy, should_stop

    def prepare_optimizer(
        self,
        model,
        optimizer_name,
        learning_rate,
        weight_decay=0.01,
        scheduler_name="linear_warmup",
        total_training_steps=None,
        warmup_ratio=0.1,):
        self.scheduler = None
        if optimizer_name == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adamW":
            no_decay = ["bias", "LayerNorm.weight"]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        if scheduler_name == "none":
                self.scheduler = None
        else:
            if total_training_steps is None:
                raise ValueError("total_training_steps must be provided when using a scheduler.")

            warmup_steps = int(total_training_steps * warmup_ratio)

            if scheduler_name == "linear_warmup":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_training_steps)
            elif scheduler_name == "cosine":
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_training_steps)
            else:
                raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return self.optimizer   

    def test(self, model, test_loader, loss_function, device, model_directory, model_name, class_names):
        checkpoint_path = os.path.join(model_directory, model_name, 'checkpoint', 'ckpt.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['net'])
            print(f"Loaded best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print("Warning: best checkpoint not found. Testing current model state.")

        model.eval()

        test_loss = 0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = loss_function(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                test_accuracy = 100.*correct/total

                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(predicted.detach().cpu().tolist())

                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            self.valid_loss_history.append(test_loss)
            self.valid_accuracy_history.append(test_accuracy)

        report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True)
        report_text = classification_report( y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()

        result_dir = os.path.join(model_directory, model_name, 'test_result')
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Test Loss: {test_loss:.6f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
            f.write(report_text)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

        with open(os.path.join(result_dir, 'test_result.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'classification_report': report_dict,
                'confusion_matrix': cm,
                'class_names': class_names,
            }, f, ensure_ascii=False, indent=2)

        with open(os.path.join(result_dir, 'test_predictions.csv'), 'w', encoding='utf-8') as f:
            f.write('y_true,y_pred,true_label,pred_label\n')
            for t, p in zip(y_true, y_pred):
                f.write(f'{t},{p},{class_names[t]},{class_names[p]}\n')

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(report_text)
        print(f"Saved test results to: {result_dir}")
        return test_loss, test_accuracy
            
class Trainer_sklearn():
    def __init__(self):
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.valid_loss_history = []
        self.valid_accuracy_history = []
        self.best_acc = 0

    def train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def valid(self, model, X_valid, y_valid, class_names, model_directory, model_name, vectorizer):
        preds = model.predict(X_valid)

        acc = accuracy_score(y_valid, preds) * 100
        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(y_valid, preds, target_names=class_names, digits=4, zero_division=0))

        if self.best_acc < acc:
            print('Saving..')
            self.best_acc = acc
            os.makedirs(os.path.join(model_directory, "checkpoint"), exist_ok=True)
            joblib.dump(
                {
                    "model": model,
                    "vectorizer": vectorizer,
                    "acc": acc,
                    "model_name": model_name,
                    "class_names": class_names,
                },
                os.path.join(model_directory, "checkpoint", f"{model_name}.joblib")
            )
            
    def test(self, model, X_test, y_test, class_names, model_directory, model_name):
        checkpoint_path = os.path.join(model_directory, "checkpoint", f"{model_name}.joblib")
        if os.path.exists(checkpoint_path):
            saved = joblib.load(checkpoint_path)
            model = saved["model"]

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        report_text = classification_report(y_test, preds, target_names=class_names, digits=4, zero_division=0)
        report_dict = classification_report(y_test, preds, target_names=class_names, digits=4, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()

        result_dir = os.path.join(model_directory, "test_result")
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "test_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"Test Accuracy: {acc:.4f}\n\n")
            f.write(report_text)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

        with open(os.path.join(result_dir, "test_result.json"), "w", encoding="utf-8") as f:
            json.dump({
                "test_accuracy": acc,
                "classification_report": report_dict,
                "confusion_matrix": cm,
                "class_names": class_names,
            }, f, ensure_ascii=False, indent=2)

        with open(os.path.join(result_dir, "test_predictions.csv"), "w", encoding="utf-8") as f:
            f.write("y_true,y_pred\n")
            for t, p in zip(y_test, preds):
                f.write(f"{t},{p}\n")

        print(f"Test Accuracy: {acc:.4f}")
        print(report_text)
        print(f"Saved test results to: {result_dir}")
        return acc
