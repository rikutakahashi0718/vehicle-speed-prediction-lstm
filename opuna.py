# 必要なライブラリをインポート
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import optuna
import os # osモジュールを追加
import plotly # Optuna可視化用
import kaleido # Optuna可視化用 (画像保存用)
import sklearn # Optuna可視化用 (パラメータ重要度計算用)

#######################################################

# CSVデータを読み込み、テンソルとして返す関数
def get_data_tensor(path):
    row_data = []
    with open(path, "r") as f:
        csv_file = csv.reader(f)
        header = next(csv_file)  # ヘッダーをスキップ
        for r in csv_file:
            r = [float(val) for val in r]
            row_data.append(torch.tensor(r))
    return torch.stack(row_data, 0)

# 特徴量の平均と標準偏差を計算
def get_features_mean_std(features):
    feat_mean = torch.mean(features, dim=0)
    feat_std = torch.std(features, dim=0)
    # 0で割ることを避けるために、標準偏差が0の場合は1e-6などの小さい値に置き換える
    feat_std[feat_std == 0] = 1e-6
    return feat_mean, feat_std

# 時系列データを使った学習用データセットクラス
class SeqDatasetTrain(Dataset):
    def __init__(self, features, n_look_back, n_pred_step):
        self.n_look_back = n_look_back
        self.n_pred_step = n_pred_step

        # featuresをコピーして、元のデータを変更しないようにする
        self.features = features.clone()

        # 特徴量の正規化
        feat_mean, feat_std = get_features_mean_std(self.features)
        for i in range(1, self.features.shape[1]):
            self.features[:, i] = (self.features[:, i]-feat_mean[i]) / feat_std[i]

        # ラベル作成（0〜85km/hを5km/h刻みで18クラスに分けてone-hotエンコード）
        n_labels = 18 # 出力クラス数は18に固定
        speeds_col = torch.clamp(torch.floor(self.features[:, 0] / 5), min=0, max=n_labels-1)
        self.labels = F.one_hot(speeds_col.to(torch.int64), num_classes=n_labels).to(torch.float)

    def __len__(self):
        # データセットの長さが負にならないようにチェック
        length = self.features.shape[0] - self.n_look_back - self.n_pred_step
        return max(0, length)

    def __getitem__(self, index):
        return self.features[index:index+self.n_look_back], self.labels[index+self.n_look_back+self.n_pred_step]

# テスト用データセットクラス（正規化パラメータは学習データから受け取る）
class SeqDatasetTest(Dataset):
    def __init__(self, features, n_look_back, n_pred_step, feat_mean, feat_std):
        self.n_look_back = n_look_back
        self.n_pred_step = n_pred_step

        # featuresをコピーして、元のデータを変更しないようにする
        self.features = features.clone()

        for i in range(1, self.features.shape[1]):
            self.features[:, i] = (self.features[:, i]-feat_mean[i]) / feat_std[i]

        n_labels = 18 # 出力クラス数は18に固定
        speeds_col = torch.clamp(torch.floor(self.features[:, 0] / 5), min=0, max=n_labels-1)
        self.labels = F.one_hot(speeds_col.to(torch.int64), num_classes=n_labels).to(torch.float)

    def __len__(self):
        # データセットの長さが負にならないようにチェック
        length = self.features.shape[0] - self.n_look_back - self.n_pred_step
        return max(0, length)

    def __getitem__(self, index):
        return self.features[index:index+self.n_look_back], self.labels[index+self.n_look_back+self.n_pred_step]

# データの読み込み・分割関数
def get_train_test_dataset(path, test_ratio, n_look_back, n_pred_step):
    all_data = get_data_tensor(path)
    n_train = int(all_data.shape[0] * (1 - test_ratio))
    train_features = all_data[:n_train]
    test_features = all_data[n_train:]
    feat_mean, feat_std = get_features_mean_std(train_features)
    train_dataset = SeqDatasetTrain(train_features, n_look_back, n_pred_step)
    test_dataset = SeqDatasetTest(test_features, n_look_back, n_pred_step, feat_mean, feat_std)
    return train_dataset, test_dataset, feat_mean, feat_std

###########################################################################

# LSTMベースのニューラルネットワークモデル (新しいモデル定義)
class VietnamModel(nn.Module):
    def __init__(self, n_seq, hidden_size, num_layers, lstm_dropout,
                 fc1_out_features, fc2_out_features, fc3_out_features, fc_dropout_rate):
        super(VietnamModel, self).__init__()

        # LSTM層
        # num_layersが1の場合、dropoutは無視されるため、num_layers > 1 の場合のみdropoutを適用
        lstm_dropout_param = lstm_dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout_param)

        # 全結合層 (FC層)
        self.fc1 = nn.Linear(n_seq * hidden_size, fc1_out_features)
        self.dropout1 = nn.Dropout(fc_dropout_rate)
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)
        self.dropout2 = nn.Dropout(fc_dropout_rate)
        self.fc3 = nn.Linear(fc2_out_features, fc3_out_features)
        self.dropout3 = nn.Dropout(fc_dropout_rate)
        self.fc4 = nn.Linear(fc3_out_features, 18) # 出力層は18クラス

    def forward(self, x):
        feat, _ = self.lstm(x) # LSTM通過
        feat = feat.reshape(feat.shape[0], -1) # Flatten

        feat = torch.relu(self.fc1(feat)) # ReLU活性化関数
        feat = self.dropout1(feat)

        feat = torch.relu(self.fc2(feat)) # ReLU活性化関数
        feat = self.dropout2(feat)

        feat = torch.relu(self.fc3(feat)) # ReLU活性化関数
        feat = self.dropout3(feat)

        out = self.fc4(feat) # 最終出力
        return out

###############################################################################

# モデルの学習関数 (Optuna用に修正)
def train(model, train_loader, test_loader, epochs, device, optimizer, criterion, scheduler, trial=None):
    # 学習ログ保存用辞書
    log_dict = {
        "train_acc_list": list(),
        "train_loss_list": list(),
        "test_acc_list": list(),
        "test_loss_list": list()
    }

    model.to(device)
    for epoch in range(epochs):
        # --- 学習 ---
        train_acc = list()
        train_loss = list()
        model.train()
        for input, label in train_loader:
            input = input.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output_train = model(input)

            correct = torch.sum(torch.argmax(output_train, 1) == torch.argmax(label, 1)).item()
            train_acc.append(correct / input.shape[0])
            loss = criterion(output_train, label)
            train_loss.append(loss.item() / input.shape[0])

            loss.backward()
            optimizer.step()

        scheduler.step()

        avg_train_acc = np.mean(train_acc)
        avg_train_loss = np.mean(train_loss)
        log_dict["train_acc_list"].append(avg_train_acc)
        log_dict["train_loss_list"].append(avg_train_loss)

        # --- テスト ---
        result, test_acc, test_loss = test(model, test_loader, criterion, device)
        log_dict["test_acc_list"].append(test_acc)
        log_dict["test_loss_list"].append(test_loss)

        # OptunaのEarly Stopping
        if trial is not None:
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return model, log_dict

# モデルの評価関数
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

# 学習・テスト結果をプロットする関数
def plot_graph(_dir, name, _dict):
    fig = plt.figure()
    plt.plot(_dict[name])
    plt.title(name)
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.join(_dir, 'resultsplots'), exist_ok=True)
    fig.savefig(os.path.join(_dir, 'resultsplots', '{}.png'.format(name)))
    plt.close(fig) # メモリ解放のためプロットを閉じる

###########################################################################
# Optunaの目的関数
def objective(trial):
    # n_pred_stepは15で固定
    N_PRED_STEP = 15

    # ハイパーパラメータの定義 (すべてのパラメータをOptunaで最適化)
    lr = trial.suggest_loguniform('lr', 1e-7, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-7, 1e-1)

    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256, 512]) # 探索範囲を拡大
    num_layers = trial.suggest_int('num_layers', 1, 5) # 探索範囲を拡大
    lstm_dropout = trial.suggest_uniform('lstm_dropout', 0.0, 0.5) # LSTM専用ドロップアウト

    fc1_out_features = trial.suggest_categorical('fc1_out_features', [128, 256, 512, 1024])
    fc2_out_features = trial.suggest_categorical('fc2_out_features', [128, 256, 512, 1024])
    fc3_out_features = trial.suggest_categorical('fc3_out_features', [64, 128, 256, 512, 1024]) # 新しいFC層の出力サイズ
    fc_dropout_rate = trial.suggest_uniform('fc_dropout_rate', 0.1, 0.5) # FC層ドロップアウト

    epochs = trial.suggest_categorical('epochs', [25, 50, 75, 100, 120]) # [25, 50, 75, 100, 120]
    train_batch_size = trial.suggest_categorical('train_batch_size', [32, 64, 128, 256, 512, 1024])
    test_batch_size = trial.suggest_categorical('test_batch_size', [32, 64, 128, 256, 512, 1024])
    n_look_back = trial.suggest_categorical('n_look_back', [30, 60])

    # データ読み込み・分割
    data_path = r'my_car_driving_data/mydrivingdata_everything_combined_updated.csv'
    train_dataset, test_dataset, feat_mean, feat_std = get_train_test_dataset(
        path=data_path, test_ratio=0.2, n_look_back=n_look_back, n_pred_step=N_PRED_STEP # 固定値を使用
    )

    # データセットのサイズが0になる可能性があるため、チェックを追加
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print(f"Warning: Dataset is too small for n_look_back={n_look_back} and n_pred_step={N_PRED_STEP}. Pruning trial.")
        return 0.0 # あるいはoptuna.exceptions.TrialPruned()を発生させる

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    # モデルとデバイス設定 (全てのハイパーパラメータを引数で渡す)
    model = VietnamModel(
        n_seq=n_look_back,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lstm_dropout=lstm_dropout, # 新しい引数
        fc1_out_features=fc1_out_features,
        fc2_out_features=fc2_out_features,
        fc3_out_features=fc3_out_features, # 新しい引数
        fc_dropout_rate=fc_dropout_rate # 新しい引数
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 最適化と損失関数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) # gammaは固定
    criterion = nn.CrossEntropyLoss()

    # 学習
    model, log_dict = train(model, train_loader, test_loader, epochs, device, optimizer, criterion, scheduler, trial)

    # 最終的なテスト精度を目的関数の値として返す
    final_test_acc = log_dict["test_acc_list"][-1]
    return final_test_acc

# Optunaのスタディを作成し、最適化を実行
if __name__ == '__main__':
    study = optuna.create_study(direction='maximize') # 新しいスタディを毎回作成する場合
    study.optimize(objective, n_trials=50) # 試行回数を指定

    print("\nNumber of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 最適化されたモデルの学習ログをプロット (best_paramsの取得まで)
    best_params = study.best_params
    N_PRED_STEP_FIXED = 15 # n_pred_stepは固定値

    # 最適なハイパーパラメータでデータローダーを再構築
    best_n_look_back = best_params['n_look_back']
    best_train_batch_size = best_params['train_batch_size']
    best_test_batch_size = best_params['test_batch_size']
    best_epochs = best_params['epochs']

    data_path = r'my_car_driving_data/mydrivingdata_everything_combined_updated.csv'
    train_dataset, test_dataset, feat_mean, feat_std = get_train_test_dataset(
        path=data_path, test_ratio=0.2, n_look_back=best_n_look_back, n_pred_step=N_PRED_STEP_FIXED
    )

    train_loader = DataLoader(train_dataset, batch_size=best_train_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=best_test_batch_size, shuffle=False, num_workers=0)

    # 最適なハイパーパラメータでモデルを初期化 (全てのハイパーパラメータを引数で渡す)
    model_optimized_for_log = VietnamModel(
        n_seq=best_n_look_back,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        lstm_dropout=best_params['lstm_dropout'],
        fc1_out_features=best_params['fc1_out_features'],
        fc2_out_features=best_params['fc2_out_features'],
        fc3_out_features=best_params['fc3_out_features'],
        fc_dropout_rate=best_params['fc_dropout_rate']
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    optimizer_optimized_for_log = optim.Adam(model_optimized_for_log.parameters(),
                                             lr=best_params['lr'],
                                             weight_decay=best_params['weight_decay'])
    scheduler_optimized_for_log = optim.lr_scheduler.ExponentialLR(optimizer_optimized_for_log, gamma=0.8)
    criterion_optimized_for_log = nn.CrossEntropyLoss()

    print("\nTraining model with best parameters for logging and plotting...")
    model_optimized_for_log, log_dict_optimized = train(
        model_optimized_for_log, train_loader, test_loader,
        best_epochs, device, optimizer_optimized_for_log,
        criterion_optimized_for_log, scheduler_optimized_for_log
    )

    # 学習経過をプロット
    print("Generating training process plots for the best trial...")
    plot_graph("", "train_acc_list", log_dict_optimized)
    plot_graph("", "train_loss_list", log_dict_optimized)
    plot_graph("", "test_acc_list", log_dict_optimized)
    plot_graph("", "test_loss_list", log_dict_optimized)
    print("Training process plots saved to 'resultsplots' directory.")

    """   # Optunaの可視化
    print("Generating optimization plots...")
    plot_save_dir = "optuna_plots_new_new_version2"
    os.makedirs(plot_save_dir, exist_ok=True)

    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_image(os.path.join(plot_save_dir, "neweverything_optimization_history.png"))
    # plt.close(fig_history) # Plotly Figureは閉じない

    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_image(os.path.join(plot_save_dir, "neweverything_param_importances.png"))
    # plt.close(fig_importance) # Plotly Figureは閉じない

    fig_parallel = optuna.visualization.plot_parallel_coordinate(study)

    fig_parallel.update_layout(
        width=2400,  # 幅を大きくする
        height=1200,  # 高さを大きくする
        # font=dict(size=10)
    )

    fig_parallel.write_image(os.path.join(plot_save_dir, "neweverything_parallel_coordinate.png"))
    # plt.close(fig_parallel) # Plotly Figureは閉じない

    try:
        fig_contour_lr_wd = optuna.visualization.plot_contour(study, params=['lr', 'weight_decay'])
        fig_contour_lr_wd.write_image(os.path.join(plot_save_dir, "neweverything_contour_lr_weight_decay.png"))
        # plt.close(fig_contour_lr_wd)
    except ValueError as e:
        print(f"Could not generate contour plot for 'lr' and 'weight_decay'. Error: {e}")

    try:
        # 新たに追加されたfc3_out_featuresを含めてプロット例を調整
        fig_contour_hs_fc1 = optuna.visualization.plot_contour(study, params=['hidden_size', 'fc1_out_features'])
        fig_contour_hs_fc1.write_image(os.path.join(plot_save_dir, "neweverything_contour_hidden_fc1.png"))
        # plt.close(fig_contour_hs_fc1)
    except ValueError as e:
        print(f"Could not generate contour plot for 'hidden_size' and 'fc1_out_features'. Error: {e}")

    try:
        # 新しいFC層とドロップアウトの組み合わせもプロット可能
        fig_contour_fc_dropout = optuna.visualization.plot_contour(study, params=['fc2_out_features', 'fc_dropout_rate'])
        fig_contour_fc_dropout.write_image(os.path.join(plot_save_dir, "neweverything_contour_fc2_dropout.png"))
    except ValueError as e:
        print(f"Could not generate contour plot for 'fc2_out_features' and 'fc_dropout_rate'. Error: {e}")

    try:
        fig_contour_lr_wd = optuna.visualization.plot_contour(study, params=['lstm_dropout', 'weight_decay'])
        fig_contour_lr_wd.write_image(os.path.join(plot_save_dir, "neweverything_contour_lstm_dropout_weight_decay.png"))
        # plt.close(fig_contour_lr_wd)
    except ValueError as e:
        print(f"Could not generate contour plot for 'lr' and 'weight_decay'. Error: {e}")

    try:
        fig_contour_lr_wd = optuna.visualization.plot_contour(study, params=['lstm_dropout', 'fc_dropout_rate'])
        fig_contour_lr_wd.write_image(os.path.join(plot_save_dir, "neweverything_contour_lstm_dropout_fc_dropout_rate.png"))
        # plt.close(fig_contour_lr_wd)
    except ValueError as e:
        print(f"Could not generate contour plot for 'lr' and 'weight_decay'. Error: {e}")

    try:
        fig_contour_lr_wd = optuna.visualization.plot_contour(study, params=['n_look_back', 'fc_dropout_rate'])
        fig_contour_lr_wd.write_image(os.path.join(plot_save_dir, "neweverything_contour_n_look_back_fc_dropout_rate.png"))
        # plt.close(fig_contour_lr_wd)
    except ValueError as e:
        print(f"Could not generate contour plot for 'lr' and 'weight_decay'. Error: {e}")

    try:
        fig_contour_lr_wd = optuna.visualization.plot_contour(study, params=['n_look_back', 'lstm_dropout'])
        fig_contour_lr_wd.write_image(os.path.join(plot_save_dir, "neweverything_contour_n_look_back_lstm_dropout_rate.png"))
        # plt.close(fig_contour_lr_wd)
    except ValueError as e:
        print(f"Could not generate contour plot for 'lr' and 'weight_decay'. Error: {e}")

    try:
        fig_contour_lr_wd = optuna.visualization.plot_contour(study, params=['n_look_back', 'weight_decay'])
        fig_contour_lr_wd.write_image(os.path.join(plot_save_dir, "neweverything_contour_n_look_back_weight_decay_rate.png"))
        # plt.close(fig_contour_lr_wd)
    except ValueError as e:
        print(f"Could not generate contour plot for 'lr' and 'weight_decay'. Error: {e}")

    try:
        fig_contour_lr_wd = optuna.visualization.plot_contour(study, params=['n_look_back', 'lr'])
        fig_contour_lr_wd.write_image(os.path.join(plot_save_dir, "neweverything_contour_n_look_back_lr_rate.png"))
        # plt.close(fig_contour_lr_wd)
    except ValueError as e:
        print(f"Could not generate contour plot for 'lr' and 'weight_decay'. Error: {e}")

    try:
        fig_contour_lr_wd = optuna.visualization.plot_contour(study, params=['n_look_back', 'hidden_size'])
        fig_contour_lr_wd.write_image(os.path.join(plot_save_dir, "neweverything_contour_n_look_back_hidden_size_rate.png"))
        # plt.close(fig_contour_lr_wd)
    except ValueError as e:
        print(f"Could not generate contour plot for 'lr' and 'weight_decay'. Error: {e}")

    fig_slice = optuna.visualization.plot_slice(study)

    for i in range(len(fig_slice.data)):
        xaxis_name = f'xaxis{i+1}' if i > 0 else 'xaxis'
        if xaxis_name in fig_slice.layout:
            fig_slice.layout[xaxis_name].tickangle = -45  # ラベルを-45度回転
            fig_slice.layout[xaxis_name].tickfont = dict(size=8) # フォントサイズを8に設定 (適宜調整)
            # 必要であれば、軸タイトル自体のフォントサイズも調整
            # fig_slice.layout[xaxis_name].title.font = dict(size=10)

    # グラフ全体の幅を調整してスペースを確保するのも有効
    fig_slice.update_layout(
        width=2400, # 例: 幅を広げる場合。ハイパーパラメータの数によって調整
        height=1200  # 例: 高さを調整する場合
    )

    fig_slice.write_image(os.path.join(plot_save_dir, "neweverything_slice_plot.png"))
    # plt.close(fig_slice) # Plotly Figureは閉じない
    print(f"Optimization plots saved to '{plot_save_dir}' directory.")
    """