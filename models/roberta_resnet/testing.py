import pandas as pd
import os
import numpy as np
import torch
import random
import functools
import operator
import cv2
import collections
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel, VisualBertModel, VisualBertConfig, RobertaTokenizer, RobertaModel
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from dataloader import meme_dataset
import json
import warnings
import codecs
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import augly.text as txtaugs
import augly.image as imaugs


# TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
DROPOUT = 0.2
HIDDEN_SIZE = 128
BATCH_SIZE = 7
NUM_LABELS = 2
NUM_EPOCHS = 20 #50
EARLY_STOPPING = {"patience": 30, "delta": 0.005}
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
WEIGHT_DECAY = 0.1 
WARMUP = 0.06 
INPUT_LEN = 768
VIS_OUT = 2048
# VIS_OUT = 1280
criterion = nn.CrossEntropyLoss().cuda()

EXP_NAME= 'test2'

def salt_pepper_noise(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.4
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[coords] = 1

      # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    return out

def speckle(image):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    noisy = np.clip(noisy,0,255).astype(np.uint8)
    return noisy

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


# def validation(dl,model):
#     fin_targets=[]
#     fin_outputs=[]
#     single_labels = []
#     img_names = []
#     for i, data in enumerate(dl):
#         data['image'] = data['image'].cuda()
#         for key in data['text'].keys():
#             data['text'][key] = data['text'][key].squeeze().cuda()
#         data['slabel'] = data['slabel'].cuda()
#         with torch.no_grad():
#             predictions, merged, _ , _ = model(data['image'],data['text'], None)
#             predictions_softmax = nn.Softmax(dim=1)(predictions)
# #             print(predictions_softmax,data['slabel'])
#             outputs = predictions.argmax(1, keepdim=True).float()
#             fin_targets.extend(data['slabel'].squeeze().cpu().detach().numpy().tolist())
#             fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
#             single_labels.extend(data['slabel'])
#             img_names.extend(data['img_info'])
#     return fin_targets, fin_outputs, single_labels, img_names



def validation(dl, model):
    fin_outputs = []
    img_names = []
    with torch.no_grad():
        for i, data in enumerate(dl):
            data['image'] = data['image'].cuda()
            for key in data['text'].keys():
                data['text'][key] = data['text'][key].squeeze().cuda()
            predictions, _, _, _ = model(data['image'], data['text'], None)
            predictions_softmax = nn.Softmax(dim=1)(predictions)
            outputs = predictions.argmax(1, keepdim=True).float()
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            img_names.extend(data['img_info'])
    return fin_outputs, img_names


def write_test_results(outputs, image_names):
    dict_single = {}
    for i in range (len(image_names)):
        image_name = image_names[i]
        pred = str(int(outputs[i][0]))
        dict_single[image_name] = pred
    dict_single = collections.OrderedDict(sorted(dict_single.items()))
    json_object = json.dumps(dict_single, indent = 4)
    json_file_name = 'preds/' + EXP_NAME  + '.json'
    with open(json_file_name, "w") as outfile:
        outfile.write(json_object)

        
def get_torch_dataloaders(dataset_name,global_path, imga, texta):
    test_dataset = meme_dataset(dataset_name, 'test', TOKENIZER, imga, texta)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    return test_dataloader
        

def main():
    import sys
    global_path = '../datasets'
    # attacks = ['spread_1','spread_3','newsprint','s&p','s&p0.4','blur_text_5','s&p_text_0.2' , 'with_sp_5px', 'without_sp_5px']
    # clean_attack_names = {'ocr':'orig_ocr', 's&p0.4':'s&p_0.4'}
    # dict_zero_shot = {}
    dataset_name = 'mami'
    dict_text_adv = {}
    dict_adv = {}
    f = open('../../attack_results_contrastive.tsv', 'a')
    cont_model = ['_contrastive', '_contrastive_img_text_temp_0.5', '_contrastive_img_text_0.07']
    cont_model  = ['pass']
    for contm in cont_model:
        #for i,attack in enumerate(attacks):
            dict_results = {}
            state_dict = torch.load('newly_initialized_weights.pt')
            model = CNN_roberta_Classifier(VIS_OUT, INPUT_LEN, DROPOUT, HIDDEN_SIZE,NUM_LABELS).cuda()
            #model.load_state_dict(torch.load('saved/'+dataset_name+'.pth'))
            #model.load_state_dict(torch.load('saved/'+dataset_name+'.pth'))
            checkpoint = torch.load('saved/' + 'mami' + '_random98' + '.pth')
            model.load_state_dict(checkpoint['model_state_dict'], state_dict)
            model.eval()
            sub_data = dataset_name
            txta = None
            imga = None
            test_dataloader = get_torch_dataloaders(sub_data,global_path,imga,txta)
            outputs,  img_names = validation(test_dataloader,model)
            write_test_results(outputs, img_names)
            # '''
            # with codecs.open(str(dataset_name)+'_'+str(contm)+'_'+str(attack)+'_errorAnalysis.tsv', 'w', 'utf-8') as ea_obj:
            #     for i, each_img in enumerate(img_names):
            #         ea_obj.write("%s\t%s\t%s\n" %(each_img, str(targets[i]), str(outputs[i])))
            # '''
            # for i, img in enumerate(img_names):
            #     if img == 'covid_memes_5427.png':
            #         print(attack)
            #         # print(targets[i])
            #         print(outputs[i])
            #
            # f1_score_macro = f1_score(targets, outputs, average='macro')
            # accuracy = accuracy_score(targets, outputs)
            # print ("Final F1 score on test set: ", dataset_name, f1_score_macro)
            # print("Final Accuracy on test set: ", dataset_name, accuracy)
        #     if str(attack) in  clean_attack_names.keys():
        #         new_attack = clean_attack_names[str(attack)]
        #     else:
        #         new_attack = str(attack)
        #     #f.write(dataset_name + '\t' + new_attack + '\t' + str(sys.argv[2])  + '\t' + str(contm)  + '\t' + str(0.0) + '\t' + str(f1_score_macro) + '\n')
        #     # dict_adv[attack]=f1_score_macro
        #print(dict_text_adv,dict_adv)

if __name__ == "__main__":
    main()











