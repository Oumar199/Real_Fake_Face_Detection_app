
from fake_face_detection.utils.compute_weights import compute_weights
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import numpy as np
import torch
import os

class LionCheetahDataset(Dataset):

    def __init__(self, lion_path: str, cheetah_path: str, id_map: dict, transformer, **transformer_kwargs):
        
        # let us recuperate the transformer
        self.transformer = transformer
        
        # let us recuperate the transformer kwargs
        self.transformer_kwargs = transformer_kwargs
        
        # let us load the images 
        lion_images = glob(os.path.join(lion_path, "*"))
        
        cheetah_images = glob(os.path.join(cheetah_path, "*"))
        
        # recuperate rgb images
        self.lion_images = []
        
        self.cheetah_images = []
        
        for lion in lion_images:
            
            try:
                
                with Image.open(lion) as img:
                    
                    # let us add a transformation on the images
                    if self.transformer:
                        
                        image = self.transformer(img, **self.transformer_kwargs)
                
                self.lion_images.append(lion)
            
            except Exception as e:
                
                pass
            
        for cheetah in cheetah_images:
            
            try:
                
                with Image.open(cheetah) as img:
                    
                    # let us add a transformation on the images
                    if self.transformer:
                        
                        image = self.transformer(img, **self.transformer_kwargs)
                
                self.cheetah_images.append(cheetah)
            
            except Exception as e:
                
                pass
        
        self.images = self.lion_images + self.cheetah_images
        
        # let us recuperate the labels
        self.lion_labels = [int(id_map['lion'])] * len(self.lion_images)
        
        self.cheetah_labels = [int(id_map['cheetah'])] * len(self.cheetah_images)
        
        self.labels = self.lion_labels + self.cheetah_labels
        
        # let us recuperate the weights
        self.weights = torch.from_numpy(compute_weights(self.labels))
        
        # let us recuperate the length
        self.length = len(self.labels)
        
    def __getitem__(self, index):
        
        # let us recuperate an image
        image = self.images[index]
        
        with Image.open(image) as img:
            
            # let us recuperate a label
            label = self.labels[index]
            
            # let us add a transformation on the images
            if self.transformer:
                
                image = self.transformer(img, **self.transformer_kwargs)
                
        # let us add the label inside the obtained dictionary
        image['labels'] = label
        
        return image    
        
    def __len__(self):
        
        return self.length
        
