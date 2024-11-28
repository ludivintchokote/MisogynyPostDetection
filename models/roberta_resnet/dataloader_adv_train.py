import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json
import random
import os
import augly.text as txtaugs
import augly.image as imaugs

# import nlpaug.augmenter.char as nac
# import nlpaug.augmenter.word as naw
# import nlpaug.augmenter.sentence as nas
# import nlpaug.flow as nafc

# from nlpaug.util import Action

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from statistics import mode
import easyocr


class meme_dataset(Dataset):
    def __init__(self, dataset_name, split, tokenizer, imga, texta, image_size: int=299, pad_length: int=100):
        self.split = split            
        self.img_attack = imga
        self.text_attack = texta
        self.global_path = '../../datasets/'
        split_file = os.path.join(self.global_path,dataset_name,'files/'+self.split+'.json')
        # split_file = os.path.join(self.global_path, dataset_name,'data/' + self.split + '.jsonl')
        with open(split_file,'r') as f:
            self.data = json.load(f)
#         self.img_path = os.path.join(self.global_path,dataset_name,'img')
#         self.img_path = os.path.join(self.global_path,dataset_name,'newsprint')    
#         if not (self.img_attack=='original_text' or self.img_attack=='ocr'):
#             self.img_path = os.path.join(self.global_path,dataset_name,self.img_attack)
#         else:
        self.img_path = os.path.join(self.global_path,dataset_name,'img')
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.pad_length = pad_length
        # (500, 500)
        self.transform = transforms.Compose([transforms.Resize((500,500)),transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
         
        self.NORMALIZE_LIST = ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number']
        self.ANNOTATE_LIST = ['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored']
        self.text_processor = TextPreProcessor(
            normalize= self.NORMALIZE_LIST,
            annotate= self.ANNOTATE_LIST,
            fix_html=True,
            segmenter="twitter", 
            unpack_hashtags=True,  
            unpack_contractions=True,  
            spell_correct_elong=True,  
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )

        self.ocr = easyocr.Reader(['en']) 
    def __len__(self):
        # n = 5  # Set this to the desired number of examples
        # return min(len(self.data), n)
         return (len(self.data))
    
    
    def get_face_text(self,image_path):
        if image_path not in self.face_data:
            text_face = "no humans"
        else:
            text_face = ""
            for item in self.face_data[image_path]:
                sentence = str(item["race"]) + " " + str(item["gender"]) + " " + str(item["age"]) + " "
                text_face += sentence
        return text_face
    

    
    def get_web_text(self,image_name):
        web_path = os.path.join(self.web_path,image_name)
        web_path += '.json'
        with open (web_path,'rb') as f:
            web_file = json.load(f)
        list_web = []
        list_web.append(web_file['best_guess'][0])
        for entity in web_file['web_entities']:
            list_web.append(entity[1])

        text_web = " ".join(list_web)
        return text_web
    
    
    
    def fix_image(self,image):
        if len(np.array(image).shape) == 2:
            image = np.array(image)
            image = np.stack([image,image,image],axis=2)
            image = Image.fromarray(image)
        elif np.array(image).shape[2] == 4:
            image = np.array(image)
            image = image[:,:,:3]
            image = Image.fromarray(image)
        else:
            image = np.array(image)
            image = Image.fromarray(image)
        return image
    
    def get_image(self,path,rand_idx):
        
        image_transform = imaugs.OneOf(
                [imaugs.Blur(), imaugs.RandomNoise(), imaugs.ShufflePixels(), imaugs.ColorJitter(), imaugs.OverlayStripes()]
            )
        image = Image.open(path)
        image = self.fix_image(image)
#         if self.split == 'train' and rand_idx>1:
# #             print('attacking image')
#             image = image_transform(image)
        image = self.transform(image)
        return image
    
    
    def get_labels(self,tweet_id):
        label = mode(self.gt[tweet_id]['labels'])
        if label>0:
            label = 1
        else:
            label = 0
        return label
    
    
    def __getitem__(self, i):
        rand_idx = random.randrange(10)
        image_path = os.path.join(self.img_path,self.data[i]['img'])
        image = self.get_image(image_path,rand_idx)
        
        text_string = 'text'
#         if not (self.text_attack=='original_text' or self.text_attack=='ocr'):
#             text_string += ('_'+self.text_attack)
#         elif self.text_attack=='ocr':
#             text_string += '_ocr'

        tweet = self.data[i][text_string]
        list_corrected_tweet = self.text_processor.pre_process_doc(tweet)
        text_tweet = ' '.join(list_corrected_tweet)
        text = text_tweet
        text_transform = txtaugs.OneOf(
                [txtaugs.ChangeCase(granularity='char'), 
                 txtaugs.ReplaceBidirectional(), 
                 txtaugs.ReplaceSimilarChars(), 
                 txtaugs.SimulateTypos()]
            )
        if rand_idx>1 and self.split=='train':
            # print ('attacking text')
            text = text_transform(text)
        # else:
        #      print(text)
        if type(text)==list:
            text = text[0]
#         print(text)
        encoded = self.tokenizer.encode_plus(
            text=text,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=100,  # maximum length of a sentence
            pad_to_max_length=True,
            return_tensors='pt',  # ask the function to return PyTorch tensors
            truncation=True,
            return_attention_mask=True,
        )

#         print(encoded)

        single_label= self.data[i]['label']

        sample = {'image': image, 'text':{
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']  # Include attention mask
            }, 'slabel': single_label, 'img_info': self.data[i]['img']}
        return sample