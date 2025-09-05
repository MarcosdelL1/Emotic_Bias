import ast
import numpy as np 
import os 
from PIL import Image

import torch 
from torch.utils.data import Dataset
from torchvision import transforms


class Emotic_PreDataset(Dataset):
  ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''
  def __init__(self, x_context, x_body, y_cat, y_cont, transform, context_norm, body_norm):
    super(Emotic_PreDataset,self).__init__()
    self.x_context = x_context
    self.x_body = x_body
    self.y_cat = y_cat 
    self.y_cont = y_cont
    self.transform = transform 
    self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])  # Normalizing the context image with context mean and context std
    self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])           # Normalizing the body image with body mean and body std

  def __len__(self):
    return len(self.y_cat)
  
  def __getitem__(self, index):
    image_context = self.x_context[index]
    image_body = self.x_body[index]
    cat_label = self.y_cat[index]
    cont_label = self.y_cont[index]
    return self.context_norm(self.transform(image_context)), self.body_norm(self.transform(image_body)), torch.tensor(cat_label, dtype=torch.float32), torch.tensor(cont_label, dtype=torch.float32)/10.0


class Emotic_CSVDataset(Dataset):
  ''' Custom Emotic dataset class. Use csv files and generated data at runtime. '''
  def __init__(self, data_df, cat2ind, transform, context_norm, body_norm, data_src = './'):
    super(Emotic_CSVDataset,self).__init__()
    self.data_df = data_df
    self.data_src = data_src 
    self.transform = transform 
    self.cat2ind = cat2ind
    self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])  # Normalizing the context image with context mean and context std
    self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])           # Normalizing the body image with body mean and body std

  def __len__(self):
    return len(self.data_df)
  
  def __getitem__(self, index):
    row = self.data_df.loc[index]
    image_context = Image.open(os.path.join(self.data_src, row['Folder'], row['Filename']))
    bbox = ast.literal_eval(row['BBox'])
    image_body = image_context.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    image_context = image_context.resize((224, 224))
    image_body = image_body.resize((128, 128))
    cat_labels = ast.literal_eval(row['Categorical_Labels'])
    cont_labels = ast.literal_eval(row['Continuous_Labels'])
    one_hot_cat_labels = self.cat_to_one_hot(cat_labels)
    return self.context_norm(self.transform(image_context)), self.body_norm(self.transform(image_body)), torch.tensor(one_hot_cat_labels, dtype=torch.float32), torch.tensor(cont_labels, dtype=torch.float32)/10.0
  
  def cat_to_one_hot(self, cat):
    """
    将情感类别列表转换为one-hot编码。

    参数:
        cat (list): 包含情感类别名称的列表，例如 ['happy', 'angry']

    返回:
        one_hot_cat (np.ndarray): 长度为26的one-hot编码数组，对应26个情感类别。
    """
    one_hot_cat = np.zeros(26)  # 创建一个长度为26的全零数组，代表所有类别初始都未被激活
    for em in cat:
      # 对于每个类别名称，查找其在类别字典中的索引，并将对应位置设为1
      one_hot_cat[self.cat2ind[em]] = 1
    return one_hot_cat  # 返回one-hot编码数组
