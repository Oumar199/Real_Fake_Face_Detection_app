
import torch
import numpy as np

def fake_face_collator(batch): 
    """The data collator for training vision transformer models on fake and real face dataset

    Args:
        batch (list): A dictionary containing the pixel values and the labels

    Returns:
        dict: The final dictionary
    """
    
    new_batch = {
        'pixel_values': [],
        'labels': []
    }
    
    for x in batch:
        
        pixel_values = torch.from_numpy(x['pixel_values'][0]) if isinstance(x['pixel_values'][0], np.ndarray) \
            else x['pixel_values'][0]
        
        new_batch['pixel_values'].append(pixel_values)
        
        new_batch['labels'].append(torch.tensor(x['labels']))
    
    new_batch['pixel_values'] = torch.stack(new_batch['pixel_values'])
    
    new_batch['labels'] = torch.stack(new_batch['labels'])
    
    return new_batch
