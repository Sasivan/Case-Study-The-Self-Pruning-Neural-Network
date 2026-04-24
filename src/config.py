import torch

class Config:
    seed = 42
    batch_size = 128
    epochs = 20
    lr = 1e-3
    lambdas = [1e-5, 1e-4, 1e-3]
    threshold = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "./data"
    output_dir = "./outputs"
    report_dir = "./report"
