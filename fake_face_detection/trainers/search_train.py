
from fake_face_detection.metrics.compute_metrics import compute_metrics
from fake_face_detection.data.collator import fake_face_collator
from transformers import Trainer, TrainingArguments, set_seed
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from typing import *
import numpy as np
import json
import os

def train(epochs: int, output_dir: str, config: dict, model: nn.Module, trainer, get_datasets: Callable, log_dir: str = "fake_face_logs", metric = 'accuracy', seed: int = 0):
    
    print("------------------------- Beginning of training")
    
    set_seed(seed)
    
    # initialize the model
    model = model()
    
    # reformat the config integer type
    for key, value in config.items():
        
        if isinstance(value, np.int32): config[key] = int(value)
    
    pretty = json.dumps(config, indent = 4)
    
    print(f"Current Config: \n {pretty}")
    
    print(f"Checkpoints in {output_dir}")
    
    # recuperate the dataset
    train_dataset, test_dataset = get_datasets(config['h_flip_p'], config['v_flip_p'], config['gray_scale_p'], config['rotation'])
    
    # initialize the arguments of the training
    training_args = TrainingArguments(output_dir,
                                      per_device_train_batch_size=config['batch_size'],
                                      evaluation_strategy='epoch',
                                      save_strategy='epoch',
                                      logging_strategy='epoch',
                                      num_train_epochs=epochs,
                                      fp16=True,
                                      save_total_limit=2,
                                      push_to_hub=False,
                                      logging_dir=os.path.join(log_dir, os.path.basename(output_dir)),
                                      load_best_model_at_end=True,
                                      learning_rate=config['lr']
                                      )
    
    # train the model
    trainer_ = trainer(
        model = model,
        args = training_args,
        data_collator = fake_face_collator,
        compute_metrics = compute_metrics,
        train_dataset = train_dataset,
        eval_dataset = test_dataset
    )
    
    # train the model
    trainer_.train()
    
    # evaluate the model and recuperate metrics
    metrics = trainer_.evaluate(test_dataset)
    
    # add metrics and config to the hyperparameter panel of tensorboard
    with SummaryWriter(os.path.join(log_dir, 'hparams')) as logger:
        
        logger.add_hparams(
            config, metrics
        )
    
    print(metrics)
    
    print("------------------------- End of training")
    # recuperate the metric to evaluate
    return metrics[f'eval_{metric}']
        
