import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os # 結果保存用のディレクトリ作成のため追加

#######################################################

#データセット
def get_data_tensor(path):
    row_data = []
    with open(path, "r") as f:
        csv_file = csv.reader(f)
        header = next(csv_file)
        for r in csv_file:
            r = [float(val) for val in r]
            row_data.append(torch.tensor(r))
    return torch.stack(row_data, 0)

#変数名追加
def get_features_mean_std(features):
    feat_mean = torch.mean(features, dim=0)
    feat_std = torch.std(features, dim=0)
    return feat_mean, feat_std

#ステップの設定（テスト用データセットのみ使用）
class SeqDatasetTest(Dataset):
    def __init__(self, features, n_look_back, n_pred_step, feat_mean, feat_std):
        self.n_look_back = n_look_back
        self.n_pred_step = n_pred_step
        self.features = features.clone() # Ensure we are working with a copy

        for i in range(1, self.features.shape[1]):
            # 標準偏差がゼロの場合のゼロ除算対策
            if feat_std[i] > 1e-6:
                self.features[:, i] = (self.features[:, i] - feat_mean[i]) / feat_std[i]
            else:
                self.features[:, i] = self.features[:, i] - feat_mean[i] # 定数の場合、平均を引くだけ

        n_labels = 18
        speeds_col = torch.clamp(torch.floor(self.features[:, 0] / 5), min=0, max=n_labels-1)
        self.labels = F.one_hot(speeds_col.to(torch.int64), num_classes=n_labels).to(torch.float)

    def __len__(self):
        return self.features.shape[0] - self.n_look_back - self.n_pred_step

    def __getitem__(self, index):
        return self.features[index:index+self.n_look_back], self.labels[index+self.n_look_back+self.n_pred_step]

###########################################################################
class VietnamModel(nn.Module):
    def __init__(self, n_seq):
        super(VietnamModel, self).__init__()

        self.lstm = nn.LSTM(input_size=4, hidden_size=512, num_layers=3, batch_first=True, dropout=0.4587763445904983)
        dropout_rate = 0.3936753333461862
        self.fc1 = nn.Linear(n_seq * 512, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(1024, 256)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(256, 18)  # 出力18クラス

    def forward(self, x):
        feat, _ = self.lstm(x)  # LSTM出力（batch, seq_len, hidden）
        feat = feat.reshape(feat.shape[0], -1)  # flatten

        feat = torch.relu(self.fc1(feat))
        feat = self.dropout1(feat)

        feat = torch.relu(self.fc2(feat))
        feat = self.dropout2(feat)

        feat = torch.relu(self.fc3(feat))
        feat = self.dropout3(feat)

        out = self.fc4(feat)
        return out

###############################################################################

# 評価方法 (test関数のみ残す)
def test(model, test_loader, criterion=nn.CrossEntropyLoss(), device="cpu"):
    model.to(device) # モデルをデバイスに移動
    model.eval() # 評価モードに設定

    test_acc = []
    test_loss = []
    result = []
    for input, label in test_loader:
        input = input.to(device)
        label = label.to(device)

        with torch.no_grad(): # 勾配計算を無効化
            output_test = model(input)
            result.extend(output_test.detach().cpu().numpy()) # CPUに移動してからNumPyに変換

            # 正解率の計算
            correct = torch.sum(torch.argmax(output_test, 1) == torch.argmax(label, 1)).item()
            test_acc.append(correct / input.shape[0])

            # ロスの計算
            loss = criterion(output_test, label)
            test_loss.append(loss.item() / input.shape[0])

    test_acc = np.mean(test_acc)
    test_loss = np.mean(test_loss)
    return result, test_acc, test_loss

def plot_graph(file_path, data, title):
    # ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.show()
    fig.savefig(file_path) # 指定されたパスに保存

###########################################################################

# 設定値
N_LOOK_BACK = 240
N_PRED_STEP = 15
MODEL_PATH = 'test.pth'  # 学習済みモデルのパス

# デバイスの決定（GPU優先）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# モデルのインスタンス化
model = VietnamModel(n_seq=N_LOOK_BACK)

# モデルのロード
# map_locationでロード時にモデルを適切なデバイスに配置
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device) # モデルを最終的に指定デバイスに移動

# テストデータ準備
# train_dataを使ってfeat_meanとfeat_stdを計算するのは、
# 学習時と同じスケールでテストデータを正規化するためです。
train_csv_path = r"my_car_driving_data/mydrivingdata_everything_combined.csv" # 学習データとして使用したデータ
test_csv_path  = r"my_car_driving_data/mydrivingdata_everything_combined.csv" # 評価したいデータ（ここでは同じものを利用）

train_data = get_data_tensor(train_csv_path)
feat_mean, feat_std = get_features_mean_std(train_data) # 平均と標準偏差は学習データから取得

test_data = get_data_tensor(test_csv_path)
test_dataset = SeqDatasetTest(test_data, N_LOOK_BACK, N_PRED_STEP, feat_mean, feat_std)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# テスト実行
print("\n--- Running Test ---")
res, acc, loss = test(model, test_loader, device=device)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss    : {loss:.4f}")

# 結果の保存ディレクトリ作成
results_dir = r"newtestresults"
os.makedirs(results_dir, exist_ok=True) # ディレクトリが存在しない場合のみ作成

# 出力結果をCSVに保存
save_res_path = os.path.join(results_dir, "test_result.csv")
res = np.asarray(res)
df_res = pd.DataFrame(res)
df_res.to_csv(save_res_path, index=False)
print(f"Test results saved to: {save_res_path}")

# 最大インデックス（予測クラス）をCSVに保存
save_max_index_path = os.path.join(results_dir, "test_max_index.csv")
max_index = np.argmax(res, axis=1)
df_max_index = pd.DataFrame(max_index)
df_max_index.to_csv(save_max_index_path, index=False)
print(f"Predicted max indices saved to: {save_max_index_path}")

# 予測クラスのプロット
plot_graph(os.path.join(results_dir, "predicted_max_index.png"), max_index, "Predicted Max Index")

print("\n--- Test process completed ---")