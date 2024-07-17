import argparse
import os
import torch
import numpy as np

from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from model.iTransformer import Model  # Ensure the path is correct

class Config:
    def __init__(self):
        self.model = 'iTransformer'
        self.use_gpu = torch.cuda.is_available()
        self.use_multi_gpu = False
        self.device_ids = [0]
        self.checkpoints = './checkpoints'
        self.seq_len = 96
        self.pred_len = 24
        self.output_attention = False
        self.use_norm = True
        self.d_model = 512
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.1
        self.class_strategy = 'sum'
        self.n_heads = 8
        self.factor = 5
        self.activation = 'relu'
        self.d_ff = 2048
        self.e_layers = 2
        self.c_out = 1
        self.learning_rate = 0.001
        self.patience = 3
        self.train_epochs = 10
        self.label_len = 48  # Example value, adjust as necessary
        self.data = 'ACN'  # Placeholder, set as needed
        self.features = 'S'  # Example value, set as needed
        self.use_amp = False  # Set to True if using mixed precision
        self.batch_size = 5
        self.root_path = './dataset/ACN/'
        self.data_path = 'finalACN.csv'
def main():
    args = Config()

    exp = Exp_Long_Term_Forecast(args)
    setting = 'long_term_forecast'

    print('Starting training...')
    exp.train(setting)

    print('Training completed! Running tests...')
    exp.test(setting)

    print('Test completed! Running predictions...')
    exp.predict(setting, load=True)

if __name__ == "__main__":
    main()
