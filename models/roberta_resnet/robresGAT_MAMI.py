import math
import pandas as pd
import os
import numpy as np
import torch
import random
import functools
import operator
import cv2
from advt.attack import DeepFool
import collections
import torchvision.models as models
import torch.nn.functional as F
# from advertorch.attacks import carlini_wagner
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel, VisualBertModel, VisualBertConfig, RobertaTokenizer, RobertaModel
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from dataloader_adv_train import meme_dataset
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")




# TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
DROPOUT = 0.2
HIDDEN_SIZE = 128 #128
BATCH_SIZE = 16 #7 #16 #8
NUM_LABELS = 2
NUM_EPOCHS = 7 #30 20 #50
EARLY_STOPPING = {"patience": 30, "delta": 0.005}
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
WEIGHT_DECAY = 0.1
WARMUP = 0.06
INPUT_LEN = 768
VIS_OUT = 2048
# VIS_OUT = 1280
criterion = nn.CrossEntropyLoss().cuda()

EXP_NAME = 'roberta_resnet_base'


class CNN_roberta_Classifier(nn.Module):
    def __init__(self, vis_out, input_len, dropout, hidden_size, num_labels):
        super(CNN_roberta_Classifier, self).__init__()
        self.lm = RobertaModel.from_pretrained('roberta-base')
        self.vm = models.resnet50(pretrained=True)
        self.vm.fc = nn.Sequential(nn.Linear(vis_out, input_len))

        embed_dim = input_len
        self.merge = torch.nn.Sequential(torch.nn.ReLU(),
                                         torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(2 * embed_dim, embed_dim))

        # self.gat = GraphAttentionLayer(8, embed_dim)
        self.gat = GraphAttentionLayer(16, embed_dim)

        self.mlp = nn.Sequential(nn.Linear(input_len, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, num_labels))

        self.image_space = nn.Sequential(nn.Linear(input_len, input_len),
                                         nn.ReLU(),
                                         nn.Linear(input_len, input_len),
                                         nn.ReLU(),
                                         nn.Linear(input_len, input_len))

        self.text_space = nn.Sequential(nn.Linear(input_len, input_len),
                                        nn.ReLU(),
                                        nn.Linear(input_len, input_len),
                                        nn.ReLU(),
                                        nn.Linear(input_len, input_len))

    def forward(self, image, text, label):
        image = self.vm(image)
        text = self.lm(**text).last_hidden_state
        text_gat = self.gat(text)
        text_gat = text_gat[:, 0, :]
        img_txt = (image, text_gat)
        img_txt = torch.cat(img_txt, dim=1)
        merged = self.merge(img_txt)
        label_output = self.mlp(merged)
        return label_output, merged, image, text_gat


class GraphAttentionLayer(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()

        # Compute Q, K, V
        query = self.query_linear(x)  # [batch_size, seq_len, hidden_size]
        key = self.key_linear(x)  # [batch_size, seq_len, hidden_size]
        value = self.value_linear(x)  # [batch_size, seq_len, hidden_size]

        # Compute attention scores
        # [batch_size, seq_len, seq_len]
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.hidden_size)

        attention_scores = self.dropout(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # [batch_size, seq_len, hidden_size]
        output = torch.bmm(attention_weights, value)

        return output

def validation(dl,model):
    fin_targets=[]
    fin_outputs=[]
    single_labels = []
    img_names = []
    for i, data in enumerate(dl):
        data['image'] = data['image'].cuda()
        for key in data['text'].keys():
            data['text'][key] = data['text'][key].squeeze().cuda()
        data['slabel'] = data['slabel'].cuda()
        with torch.no_grad():
            predictions, merged, _, _ = model(data['image'],data['text'], data['slabel'])
            predictions_softmax = nn.Softmax(dim=1)(predictions)
            # print(predictions_softmax,data['slabel'])
            outputs = predictions.argmax(1, keepdim=True).float()
            fin_targets.extend(data['slabel'].squeeze().cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            single_labels.extend(data['slabel'])
            img_names.extend(data['img_info'])
    return fin_targets, fin_outputs, single_labels, img_names


def train(train_dataloader, dev_dataloader, model, optimizer, scheduler, dataset_name):
    max_acc = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            train_total_correct = 0
            train_num_correct = 0
            train_loss_values, val_loss_values, train_acc_values, val_acc_values = [], [], [], []
            train_preds = []
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data['image'] = data['image'].cuda()
                for key in data['text'].keys():
                    #                     print(data['text'][key].shape)
                    data['text'][key] = data['text'][key].squeeze(dim=1).cuda()
                data['slabel'] = data['slabel'].cuda()
                #                 print(data['text'].shape)
                output, merged, image_shifted, text_shifted = model(data['image'], data['text'], data['slabel'])
                pred = output.argmax(1, keepdim=True).float()
                loss = criterion(output, data['slabel'])
                train_loss_values.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                train_total_correct += data['image'].shape[0]
                pred = output.argmax(1, keepdim=True).float()
                tepoch.set_postfix(loss=loss.item())
        print("loss ", sum(train_loss_values) / len(train_loss_values))
        model.eval()
        targets, outputs, slabels, img_names = validation(dev_dataloader, model)
        accuracy = accuracy_score(targets, outputs)
        f1_score_micro = f1_score(targets, outputs, average='micro')
        f1_score_macro = f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        #         print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")

        if f1_score_macro > max_acc:
            max_acc = f1_score_macro
            print("new best saving, ", max_acc)
            path = 'saved/' + dataset_name + '_random' + '.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, path)
            # torch.save(model.state_dict(), 'saved/' + dataset_name + '_random' + '.pth')

        print("Best so far, ", max_acc)


def write_test_results(outputs, image_names):
    dict_single = {}
    for i in range(len(image_names)):
        image_name = image_names[i]
        pred = str(int(outputs[i][0]))
        dict_single[image_name] = pred
    dict_single = collections.OrderedDict(sorted(dict_single.items()))
    json_object = json.dumps(dict_single, indent=4)
    json_file_name = 'preds/' + EXP_NAME + '.json'
    with open(json_file_name, "w") as outfile:
        outfile.write(json_object)


def get_torch_dataloaders(dataset_name, global_path):
    train_dataset = meme_dataset(dataset_name, 'train', TOKENIZER, None, None)
    dev_dataset = meme_dataset(dataset_name, 'val', TOKENIZER, None, None)
    test_dataset = meme_dataset(dataset_name, 'test', TOKENIZER, None, None)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    return train_dataloader, dev_dataloader, test_dataloader


def main():
    global_path = '../datasets'
    datasets = ['mami']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset_name in datasets:
        train_dataloader, dev_dataloader, test_dataloader = get_torch_dataloaders(dataset_name, global_path)
        # Load the state dictionary from the file
        state_dict = torch.load('newly_initialized_weights.pt')
        model = CNN_roberta_Classifier(VIS_OUT, INPUT_LEN, DROPOUT, HIDDEN_SIZE, NUM_LABELS).cuda()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06 * (
                    len(train_dataloader) * BATCH_SIZE * NUM_EPOCHS), num_training_steps=(1 - 0.06) * (
                    len(train_dataloader) * BATCH_SIZE * NUM_EPOCHS))
        print(" Start Training on, ", dataset_name, len(train_dataloader), len(dev_dataloader), len(test_dataloader))

        checkpoint = torch.load('saved/' + 'mami' + '_random' + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'], state_dict)

        max_acc = 0
        for epoch in range(NUM_EPOCHS):
            model.train()
            with tqdm(train_dataloader, unit="batch") as tepoch:
                train_total_correct = 0
                train_num_correct = 0
                train_loss_values, val_loss_values, train_acc_values, val_acc_values = [], [], [], []
                train_preds = []
                for data in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    data['image'] = data['image'].cuda()
                    for key in data['text'].keys():
                        #                     print(data['text'][key].shape)
                        data['text'][key] = data['text'][key].squeeze(dim=1).cuda()
                    data['slabel'] = data['slabel'].cuda()
                    #                 print(data['text'].shape)
                    output, merged, image_shifted, text_shifted = model(data['image'], data['text'], data['slabel'])
                    pred = output.argmax(1, keepdim=True).float()
                    loss = criterion(output, data['slabel'])
                    train_loss_values.append(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    train_total_correct += data['image'].shape[0]
                    pred = output.argmax(1, keepdim=True).float()
                    tepoch.set_postfix(loss=loss.item())
            print("loss ", sum(train_loss_values) / len(train_loss_values))
            model.eval()
            targets, outputs, slabels, img_names = validation(dev_dataloader, model)
            accuracy = accuracy_score(targets, outputs)
            f1_score_micro = f1_score(targets, outputs, average='micro')
            f1_score_macro = f1_score(targets, outputs, average='macro')
            print(f"Accuracy Score = {accuracy}")
            #         print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")

            if f1_score_macro > max_acc:
                max_acc = f1_score_macro
                print("new best saving, ", max_acc)
                path = 'saved/' + dataset_name + '_random' + '.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                }, path)
                # torch.save(model.state_dict(), 'saved/' + dataset_name + '_random' + '.pth')

            print("Best so far, ", max_acc)


if __name__ == "__main__":
    main()

