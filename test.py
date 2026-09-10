import torch
print(torch.version.cuda)   # PyTorch がビルドされた CUDA のバージョン
print(torch.cuda.is_available())  # GPU が使えるかどうか
