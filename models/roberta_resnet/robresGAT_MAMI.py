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
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel, VisualBertModel, VisualBertConfig, \
    RobertaTokenizer, RobertaModel
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
HIDDEN_SIZE = 128  # 128
BATCH_SIZE = 7  # 7 #16 #8
NUM_LABELS = 2
NUM_EPOCHS = 5  # 30 20 #50
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
        #         self.lm = BertModel.from_pretrained('bert-base-uncased')
        self.vm = models.resnext50_32x4d(pretrained=True)  # Use ResNeXt-50
        # self.vm = models.resnet50(pretrained=True)
        self.vm.fc = nn.Sequential(nn.Linear(vis_out, input_len))
        #         self.vm = models.efficientnet_b5(pretrained=True)
        #         self.vmlp = nn.Linear(vis_out,input_len)
        #         print(self.vm)

        embed_dim = input_len
        self.merge = torch.nn.Sequential(torch.nn.ReLU(),
                                         torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(2 * embed_dim, embed_dim))
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
        #         img_cls, image_prev = self.vm(image)
        #         image = self.vmlp(image_prev)
        image = self.vm(image)
        text = self.lm(**text).last_hidden_state[:, 0, :]
        image_shifted = image
        text_shifted = text
        img_txt = (image, text)
        img_txt = torch.cat(img_txt, dim=1)
        merged = self.merge(img_txt)
        label_output = self.mlp(merged)
        return label_output, merged, image_shifted, text_shifted


def validation(dl, model):
    fin_targets = []
    fin_outputs = []
    single_labels = []
    img_names = []
    for i, data in enumerate(dl):
        data['image'] = data['image'].cuda()
        for key in data['text'].keys():
            data['text'][key] = data['text'][key].squeeze().cuda()
        data['slabel'] = data['slabel'].cuda()
        with torch.no_grad():
            predictions, merged, _, _ = model(data['image'], data['text'], data['slabel'])
            predictions_softmax = nn.Softmax(dim=1)(predictions)
            # print(predictions_softmax,data['slabel'])
            outputs = predictions.argmax(1, keepdim=True).float()
            fin_targets.extend(data['slabel'].squeeze().cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            single_labels.extend(data['slabel'])
            img_names.extend(data['img_info'])
    return fin_targets, fin_outputs, single_labels, img_names


def generate_fgsm_attack(model, X, y, text, label, epsilon=0.1, clip_min=0., clip_max=1.):
    X_adv = X.detach().clone().requires_grad_(True)
    onehot_y = F.one_hot(y.to(X.device), num_classes=2).float()
    logits, _, _, _ = model(X_adv, text, label)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, X_adv, retain_graph=True)[0]
    X_adv = X_adv.detach()
    X_adv += epsilon * grad.sign()
    X_adv = torch.clamp(X_adv, clip_min, clip_max)
    return X_adv


def generate_pgd_attack(model, X, y, text, label, epsilon=0.1, alpha=0.01, steps=10, clip_min=0., clip_max=1.):
    X_adv = X.detach().clone().requires_grad_(True)
    onehot_y = F.one_hot(y.to(X.device), num_classes=2).float()
    for _ in range(steps):
        logits, _, _, _ = model(X_adv, text, label)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, X_adv, retain_graph=True)[0]
        X_adv = X_adv.detach()
        X_adv = X_adv.data + alpha * grad.sign()
        X_adv = torch.clamp(X_adv, clip_min, clip_max)
    return X_adv


def l0_attack(model, x, y, text, label, epsilon=0.1, alpha=0.01, gamma=0.01, max_iter=5, rand_start=True):
    """
    Performs an L0 attack on the input image `x` with label `y` using the given model.

    Args:
        model (nn.Module): The PyTorch model to attack.
        x (torch.Tensor): The input image tensor of shape (N, C, H, W).
        y (torch.Tensor): The label tensor of shape (N,).
        text (dict): The input text tensor of shape (N, seq_len).
        label (dict): The input label tensor of shape (N,).
        epsilon (int): The maximum number of pixels to perturb.
        alpha (float): The step size for the attack.
        gamma (float): The decay factor for the attack.
        max_iter (int): The maximum number of iterations for the attack.
        rand_start (bool): Whether to start the attack with a random perturbation.

    Returns:
        x_adv (torch.Tensor): The adversarial example tensor of shape (N, C, H, W).
    """
    x_adv = x.clone().detach()
    if rand_start:
        mask = torch.rand(*x.shape) < 0.5
        x_adv[mask] = 1 - x_adv[mask]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([x_adv], lr=alpha)
    for i in range(max_iter):
        optimizer.zero_grad()
        text_tensors = {k: v.unsqueeze(1) for k, v in text.items()}
        label_dict = {k: v for k, v in zip(text_tensors.keys(), label)}
        text_tensors = {k: v.squeeze(1) for k, v in text_tensors.items()}
        logits, _, _, _ = model(x_adv, text_tensors, label_dict)
        loss = loss_fn(logits, y.squeeze()) + gamma * torch.sum(torch.abs(x - x_adv))
        loss.backward()
        optimizer.step()
        if torch.sum(torch.abs(x - x_adv)) <= epsilon:
            break
    return x_adv


def generate_one_pixel_attack(model, X, y, text, label, epsilon=0.1, alpha=0.01, steps=10, clip_min=0., clip_max=1.):
    X_adv = X.detach().clone()
    onehot_y = F.one_hot(y.to(X.device), num_classes=2).float()
    for _ in range(steps):
        logits, _, _, _ = model(X_adv, text, label)
        logits_y = logits[0, y]
        logits_not_y = torch.max(logits[0, :y], logits[0, y + 1:]).values
        if logits_y > logits_not_y:
            break
        channel_idx = random.randint(0, X.shape[1] - 1)
        row_idx = random.randint(0, X.shape[2] - 1)
        col_idx = random.randint(0, X.shape[3] - 1)
        if X.shape[1] == 1:
            X_adv[0, 0, row_idx, col_idx] += alpha
        else:
            X_adv[0, channel_idx, row_idx, col_idx] += alpha
        X_adv = torch.clamp(X_adv, clip_min, clip_max)
    return X_adv


def adversarial_drop_attack(model, input_tensor, drop_prob=0.9):
    """
    Perform adversarial drop attack on the input tensor.
    """
    model.eval()
    input_tensor_clone = input_tensor.detach().clone()
    input_tensor_clone[input_tensor_clone > 0] = 1
    input_tensor_clone[input_tensor_clone <= 0] = 0
    mask = input_tensor_clone.clone()
    mask = F.dropout(mask, p=drop_prob, training=True)
    input_tensor_clone *= mask
    return input_tensor_clone


def generate_pgd_l2_attack(model, X, y, text, label, epsilon=0.1, alpha=0.01, steps=10, clip_min=0., clip_max=1.):
    X_adv = X.detach().clone().requires_grad_(True)
    onehot_y = F.one_hot(y.to(X.device), num_classes=2).float()
    for _ in range(steps):
        logits, _, _, _ = model(X_adv, text, label)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, X_adv, retain_graph=True)[0]
        X_adv = X_adv.detach()
        X_adv = X_adv.data + alpha * grad / torch.norm(grad, p=2)
        X_adv = torch.clamp(X_adv, clip_min, clip_max)
    return X_adv


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
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06 * (
                    len(train_dataloader) * BATCH_SIZE * NUM_EPOCHS), num_training_steps=(1 - 0.06) * (
                    len(train_dataloader) * BATCH_SIZE * NUM_EPOCHS))
        print(" Start Training on, ", dataset_name, len(train_dataloader), len(dev_dataloader), len(test_dataloader))
        #
        # train(train_dataloader, dev_dataloader, model, optimizer, scheduler, dataset_name)
        # Exclude the lm module's state dictionary
        # state_dict = {k: v for k, v in model.state_dict().items() if 'lm' not in k}
        # Save the state dictionary to a file
        # torch.save(state_dict, 'newly_initialized_weights.pt')
        # Load the state dictionary into the model
        # model.load_state_dict(state_dict)

        # model.load_state_dict(torch.load('saved/'+dataset_name+'_random'+'.pth'))
        # checkpoint = torch.load('saved/'+'mami'+'_randomAdvDrop'+'.pth')
        # checkpoint = torch.load('saved/' + 'mami' + '_random98' + '.pth')
        # model.load_state_dict(checkpoint['model_state_dict'], state_dict)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model.eval()
        # df = DeepFool(model.vm, max_iter=100, device=device)

        # targets, outputs, slabels, img_names = validation(dev_dataloader, model)
        # f1_score_macro = f1_score(targets, outputs, average='macro')
        # accuracy = accuracy_score(targets, outputs)
        # print("Final F1 score on validation set: ", dataset_name, f1_score_macro)
        # print("Final Accuracy on validation set: ", dataset_name, accuracy)

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

# class CNN_roberta_Classifier(nn.Module):
#     def __init__(self, vis_out, input_len, dropout, hidden_size, num_labels):
#         super(CNN_roberta_Classifier, self).__init__()
#         self.lm = RobertaModel.from_pretrained('roberta-base')
#         self.vm = models.resnet50(pretrained=True)
#         self.vm.fc = nn.Sequential(nn.Linear(vis_out, input_len))
#
#         embed_dim = input_len
#         self.merge = torch.nn.Sequential(torch.nn.ReLU(),
#                                          torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Linear(2 * embed_dim, embed_dim))
#
#         self.gat = GraphAttentionLayer(8, embed_dim)
#
#         self.mlp = nn.Sequential(nn.Linear(input_len, hidden_size),
#                                  nn.ReLU(),
#                                  nn.Linear(hidden_size, hidden_size),
#                                  nn.ReLU(),
#                                  nn.Linear(hidden_size, num_labels))
#
#         self.image_space = nn.Sequential(nn.Linear(input_len, input_len),
#                                          nn.ReLU(),
#                                          nn.Linear(input_len, input_len),
#                                          nn.ReLU(),
#                                          nn.Linear(input_len, input_len))
#
#         self.text_space = nn.Sequential(nn.Linear(input_len, input_len),
#                                         nn.ReLU(),
#                                         nn.Linear(input_len, input_len),
#                                         nn.ReLU(),
#                                         nn.Linear(input_len, input_len))
#
#     def forward(self, image, text, label):
#         image = self.vm(image)
#         text = self.lm(**text).last_hidden_state
#         text_gat = self.gat(text)
#         text_gat = text_gat[:, 0, :]
#         img_txt = (image, text_gat)
#         img_txt = torch.cat(img_txt, dim=1)
#         merged = self.merge(img_txt)
#         label_output = self.mlp(merged)
#         return label_output, merged, image, text_gat
#
#
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, num_heads, hidden_size):
#         super(GraphAttentionLayer, self).__init__()
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.query_linear = nn.Linear(hidden_size, hidden_size)
#         self.key_linear = nn.Linear(hidden_size, hidden_size)
#         self.value_linear = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         batch_size, seq_len, hidden_size = x.size()
#         query = self.query_linear(x)
#         key = self.key_linear(x)
#         value = self.value_linear(x)
#         attention_scores = torch.matmul(query, key.T) / math.sqrt(hidden_size)
#         attention_scores = self.dropout(attention_scores)
#         attention_weights = nn.functional.softmax(attention_scores, dim=-1)
#         output = torch.matmul(attention_weights, value)
#         return output
#


# for epoch in range(NUM_EPOCHS):
#     # model.train()
#     model.train()
#     with tqdm(train_dataloader, unit="batch") as tepoch:
#         train_total_correct = 0
#         train_num_correct = 0
#         train_loss_values, val_loss_values, train_acc_values, val_acc_values = [], [], [], []
#         train_preds = []
#         print('Generating adversarial examples using L0 attack for training')
#         for data in tepoch:
#             tepoch.set_description(f"Epoch {epoch}")
#             data['image'] = data['image'].cuda()
#             # data['image'] = adversarial_drop_attack(model, data['image'].cuda()).cuda()
#
#             for key in data['text'].keys():
#                 #                     print(data['text'][key].shape)
#                 data['text'][key] = data['text'][key].squeeze(dim=1).cuda()
#             data['slabel'] = data['slabel'].cuda()
#             #                 print(data['text'].shape)
#
#             # Apply DeepFool attack to the input images
#             # data['image'] = l0_attack(model,data['image'].detach(), data['slabel'],data['text'], data['slabel'])
#             # Generate adversarial examples using the L0 attack
#             x_adv = l0_attack(model, data['image'], data['slabel'], data['text'], data['slabel'], epsilon=0.1,
#                               alpha=0.01, gamma=0.01,
#                               max_iter=5, rand_start=True)
#             # Concatenate the original and adversarial examples
#             # data['image'] = torch.cat((data['image'], x_adv), dim=0)
#             # data['slabel'] = torch.cat((data['slabel'], data['slabel']), dim=0)
#             # image_adv = generate_fgsm_attack(model, data['image'].detach(), data['slabel'],data['text'], data['slabel'])
#             output, merged, image_shifted, text_shifted = model(data['image'], data['text'], data['slabel'])
#             output_adv, _, _, _ = model(x_adv, data['text'], data['slabel'])
#
#             # Calculate loss using both original and adversarial examples
#             loss = 0.5 * criterion(output, data['slabel']) + 0.5 * criterion(output_adv, data['slabel'])
#
#             # pred = output.argmax(1, keepdim=True).float()
#             # loss = criterion(output, data['slabel'])
#             train_loss_values.append(loss)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             scheduler.step()
#             train_total_correct += data['image'].shape[0]
#             # pred = 0.5*output_adv.argmax(1, keepdim=True).float()+0.5*output.argmax(1, keepdim=True).float()
#             tepoch.set_postfix(loss=loss.item())
#     print("loss ", sum(train_loss_values) / len(train_loss_values))
#     model.eval()
#     targets, outputs, slabels, img_names = validation(dev_dataloader, model)
#     accuracy = accuracy_score(targets, outputs)
#     f1_score_micro = f1_score(targets, outputs, average='micro')
#     f1_score_macro = f1_score(targets, outputs, average='macro')
#     print(f"Accuracy Score = {accuracy}")
#     #         print(f"F1 Score (Micro) = {f1_score_micro}")
#     print(f"F1 Score (Macro) = {f1_score_macro}")
#
#     if f1_score_macro > max_acc:
#         max_acc = f1_score_macro
#         print("new best saving, ", max_acc)
#         path = 'saved/' + dataset_name + '_random' + '.pth'
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': criterion,
#         }, path)
#         # torch.save(model.state_dict(), 'saved/' + dataset_name + '_random' + '.pth')
#
#     print("Best so far, ", max_acc)






