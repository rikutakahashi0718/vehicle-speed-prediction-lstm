import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os

#######################################################

#データセット
#impoert excel_data
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

#ステップの設定
class SeqDatasetTrain(Dataset):#class関数
    def __init__(self, features, n_look_back, n_pred_step):
        self.n_look_back = n_look_back
        self.n_pred_step = n_pred_step
        self.features = features

        # Calculate mean and standard deviation
        feat_mean, feat_std = get_features_mean_std(self.features)

        for i in range(1, self.features.shape[1]):
            features[:, i] = (features[:, i]-feat_mean[i]) / feat_std[i]

        n_labels = 18
        speeds_col = torch.clamp(torch.floor(self.features[:, 0] / 5), min=0, max=n_labels-1)
        self.labels = F.one_hot(speeds_col.to(torch.int64), num_classes=n_labels ).to(torch.float)

    def __len__(self):
        return self.features.shape[0] - self.n_look_back - self.n_pred_step

    def __getitem__(self, index):
        return self.features[index:index+self.n_look_back], self.labels[index+self.n_look_back+self.n_pred_step]

class SeqDatasetTest(Dataset):
    def __init__(self, features, n_look_back, n_pred_step, feat_mean, feat_std):
        self.n_look_back = n_look_back
        self.n_pred_step = n_pred_step
        self.features = features

        for i in range(1, features.shape[1]):
            features[:, i] = (features[:, i]-feat_mean[i]) / feat_std[i]

        n_labels = 18
        speeds_col = torch.clamp(torch.floor(self.features[:, 0] / 5), min=0, max=n_labels-1)
        self.labels = F.one_hot(speeds_col.to(torch.int64), num_classes=n_labels).to(torch.float)

    def __len__(self):
        return self.features.shape[0] - self.n_look_back - self.n_pred_step

    def __getitem__(self, index):
        return self.features[index:index+self.n_look_back], self.labels[index+self.n_look_back+self.n_pred_step]

#テストデータセット
def get_train_test_dataset(path, test_ratio, n_look_back, n_pred_step):
    all_data = get_data_tensor(path)
    n_train = int(all_data.shape[0] *  (1-test_ratio))
    train_features = all_data[:n_train]
    test_features = all_data[n_train:]
    feat_mean, feat_std = get_features_mean_std(train_features)
    train_dataset = SeqDatasetTrain(train_features, n_look_back, n_pred_step)
    test_dataset = SeqDatasetTest(test_features, n_look_back, n_pred_step, feat_mean, feat_std)
    return train_dataset, test_dataset

###########################################################################
class VietnamModel(nn.Module):
    def __init__(self, n_seq):
        super(VietnamModel, self).__init__()

        self.lstm = nn.LSTM(input_size=4, hidden_size=512, num_layers=4, batch_first=True, dropout=0.016652365441674973
)
        dropout_rate = 0.3894567998632305
        self.fc1 = nn.Linear(n_seq * 512, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 256)
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

###########################################################################
"""
class VietnamModel(nn.Module):

    def __init__(self, n_seq):
        super(VietnamModel, self).__init__()

        self.lstm = nn.LSTM(input_size=4, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(in_features=n_seq*64, out_features=512)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=256, out_features=18)

    def forward(self, x):
        feat, (h_n, c_n) = self.lstm(x)
        feat = self.fc1(feat.reshape(feat.shape[0], -1))
        # feat = self.fc1(h_n.squeeze(0))
        feat = torch.sigmoid(feat)
        feat = self.dropout1(feat)
        feat = self.fc2(feat)
        feat = self.dropout2(feat)
        feat = torch.sigmoid(feat)
        out = self.fc3(feat)
        return out
"""
###############################################################################

#評価方法
def train(model, train_loader, test_loader, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=0.0010249897029976085, weight_decay=4.0893013159735796e-07
)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    log_dict = {
        "train_acc_list": list(),
        "train_loss_list": list(),
        "test_acc_list": list(),
        "test_loss_list": list()
    }

    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        # Train
        train_acc = list()
        train_loss = list()
        train_list = list()
        model.train()
        for input, label in train_loader:
            input = input.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output_train = model(input)

            train_list.append(output_train.detach().to("cpu").numpy())

            correct = torch.sum(torch.argmax(output_train, 1) == torch.argmax(label, 1)).item()
            train_acc.append(correct / input.shape[0])

            loss = criterion(output_train, label)
            train_loss.append(loss.item() / input.shape[0])

            loss.backward()
            optimizer.step()

        scheduler.step()

        log_dict["train_acc_list"].append(np.mean(train_acc))
        log_dict["train_loss_list"].append(np.mean(train_loss))
        print(f"Train acc={np.mean(train_acc):.4f}, loss={np.mean(train_loss)}")


        # Evaluation
        result, test_acc, test_loss = test(model, test_loader, criterion, device)

        log_dict["test_acc_list"].append(test_acc)
        log_dict["test_loss_list"].append(test_loss)
        print(f"Test acc={np.mean(test_acc):.4f}, loss={np.mean(test_loss)}")

    return model, log_dict, train_list

def test(model, test_loader, criterion=nn.CrossEntropyLoss(), device="cpu"):
    model.eval()

    test_acc = list()
    test_loss = list()
    result = list()
    for input, label in test_loader:
        input = input.to(device)
        label = label.to(device)

        with torch.no_grad():
            output_test = model(input)
            result.extend(output_test.detach().to("cpu").numpy())

            correct = torch.sum(torch.argmax(output_test, 1) == torch.argmax(label, 1)).item()
            test_acc.append(correct / input.shape[0])

            loss = criterion(output_test, label)
            test_loss.append(loss.item() / input.shape[0])

    test_acc = np.mean(test_acc)
    test_loss = np.mean(test_loss)
    return result, test_acc, test_loss

def plot_graph(_dir, name, _dict):
    fig = plt.figure()
    plt.plot(_dict[name])
    plt.title(name)
    plt.show()
    fig.savefig(r'resultsplots\{}.png'.format(name))

###########################################################################

N_LOOK_BACK = 120
N_PRED_STEP = 15

#学習及び予測
data_path = r'my_car_driving_data/mydrivingdata_everything_combined_updated.csv'
train_dataset, test_dataset = get_train_test_dataset(path=data_path, test_ratio=0.1, n_look_back=N_LOOK_BACK, n_pred_step=15)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

model = VietnamModel(n_seq=N_LOOK_BACK)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model, log_dict, train_list = train(model, train_loader, test_loader, epochs=75, device=device)

torch.save(model.state_dict(), 'finaltest.pth') # change the name

####################################################################

# load model
model = VietnamModel(n_seq=N_LOOK_BACK)
state_dict = torch.load('finaltest.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


####################################################################

torch.save(model.state_dict(), "sample.pth")

plot_graph("", "train_acc_list", log_dict)
plot_graph("", "train_loss_list", log_dict)
plot_graph("", "test_acc_list", log_dict)
plot_graph("", "test_loss_list", log_dict)

MODEL_PATH = 'finaltest.pth'

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
test_csv_path  = r"my_car_driving_data/data7.csv" # 評価したいデータ（ここでは同じものを利用）

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
results_dir = r"finaltestdir"
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