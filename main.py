import torch
import torch.nn as nn
import argparse

import models
import datasets
import training_utils
import utils

parser = argparse.ArgumentParser(description='Imbalance data classifier for antibiotic-resistance in MIMIC-iv,-iii')
parser.add_argument('--seed', default=42, type=int,
                    help='seed')
parser.add_argument('--data_directory', default="../data", type=str,
                    help='data directory')
parser.add_argument('--train_file', default="train.csv", type=str,
                    help='train file name')
parser.add_argument('--valid_file', default="valid.csv", type=str,
                    help='valid file name')
parser.add_argument('--test_file', default="test.csv", type=str,
                    help='test file name')
parser.add_argument('--lr', default=2e-5, type=float,
                    help='learning rate')
parser.add_argument('--optimizer', default="adamW", type=str,
                    help='specify optimizer')
parser.add_argument('--epoch', default=3, type=int,
                    help='num epoch')
parser.add_argument('--model', default='kobert', type=str,
                    help='model name of models.py')
parser.add_argument('--model_directory', default="training_result", type=str,
                    help='model training result directory')
parser.add_argument('--early_stopping_patience', default=3, type=int,
                    help='stop training if validation loss does not improve for this many epochs')
parser.add_argument('--early_stopping_min_delta', default=0.0, type=float,
                    help='minimum validation loss improvement required to reset early stopping counter')

args = parser.parse_args()
utils.seed_everything(args.seed)

def run_kobert(trainer):
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader, class_names = datasets.prepare_dataset_kobert(args.data_directory, args.train_file, args.valid_file, args.test_file, args.seed)
    model = models.prepare_model(args.model, device, seed=args.seed)
    trainer.prepare_optimizer(model, args.optimizer, args.lr)
    for epoch in range(start_epoch, start_epoch+args.epoch):
        trainer.train(epoch, model, train_loader, criterion, device)
        valid_loss, valid_acc, should_stop = trainer.valid(epoch, model, valid_loader, criterion, device, args.model_directory, args.model)
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}. Best valid loss: {trainer.best_valid_loss:.6f}")
            break
        
    trainer.test(model, test_loader, criterion, device, args.model_directory, args.model, class_names)

def run_classicML(trainer):
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names, vectorizer = datasets.prepare_dataset_classicML(args.data_directory, args.train_file, args.valid_file, args.test_file)
    model = models.prepare_model(args.model, device, seed=args.seed)
    trainer.train(model, X_train, y_train)
    trainer.valid(model, X_valid, y_valid, class_names, args.model_directory, args.model, vectorizer)
    trainer.test(model, X_test, y_test, class_names, args.model_directory, args.model)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.model == 'kobert':
        criterion = nn.CrossEntropyLoss()
        trainer = training_utils.Trainer_pytorch(early_stopping_patience=args.early_stopping_patience, early_stopping_min_delta=args.early_stopping_min_delta)
        run_kobert(trainer)
    else:
        trainer = training_utils.Trainer_sklearn()
        run_classicML(trainer)