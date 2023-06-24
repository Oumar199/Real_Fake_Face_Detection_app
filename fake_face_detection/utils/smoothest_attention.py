import torch
import copy

# we want to take 0.2 of the pixel and 0.7 of the mean of the pixels around it 100 times
# we will take a size between the current pixel and the pixels around it                
def smooth_attention(attention: torch.Tensor, iters: int = 1000, threshold: float = 0.1, scale: float = 0.2, size: int = 3):
    
    # squeeze the attention
    attention = copy.deepcopy(attention.squeeze())
    
    # make 100 iterations
    for _ in range(iters):
        
        # initialize the difference
        difference = torch.full(attention.shape, torch.inf)
        
        # iterate over the pixels of the attention
        for i in range(attention.shape[0]):
            
            for j in range(attention.shape[1]):
                
                # recuperate the pixel
                pixel = attention[i, j]
                
                # recuperate the mean of the pixels around it
                mean = attention[max(0, i - size): min(attention.shape[0], i + size), max(0, j - size): min(attention.shape[1], j + size)].mean()
                
                # update the attention
                attention[i, j] = (1 - scale) * pixel + scale * mean
                
                # recuperate the difference
                difference[i, j] = abs(pixel - mean)
        
        # compare each difference with the threshold
        if (difference < threshold).all(): break
    
    # unsqueeze the attention
    attention = attention.unsqueeze(-1)
    
    # return the attention
    return attention
                
