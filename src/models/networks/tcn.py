# src/models/networks/tcn.py
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.01)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.01)
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(0.01)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_layers, num_channel, kernel_size=2, dropout=0.2, embedding_info=None):
        """
        Args:
            input_size (int): 連続変数の特徴量数
            output_size (int): 出力次元数
            n_layers (int): レイヤ層数
            num_channel (int): 各レイヤのチャネル数
            kernel_size (int): カーネルサイズ
            dropout (float): ドロップアウト率
            embedding_info (list of dict): 各カテゴリ変数の設定 
                             例: [{'num_categories': 33, 'embedding_dim': 8}, ...]
        """
        super(TCN, self).__init__()
        
        # レイヤ層数とチャネル数からリストを生成
        num_channels = [num_channel] * n_layers
            
        # カテゴリ変数のためのEmbedding層の構築
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        if embedding_info:
            for info in embedding_info:
                emb = nn.Embedding(info['num_categories'], info['embedding_dim'])
                # 金融データはスパースになりやすいため、正規分布で初期化
                nn.init.normal_(emb.weight, std=0.01)
                self.embeddings.append(emb)
                total_emb_dim += info['embedding_dim']
        # TCNバックボーンへの最終的な入力次元数
        self.total_input_size = input_size + total_emb_dim
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.total_input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x_cont, x_cat=None):
        """
        Args:
            x_cont: 連続変数テンソル [Batch, SeqLen, InputSize]
            x_cat: カテゴリ変数テンソル [Batch, SeqLen, NumCatFeatures] (整数型)
        """
        # カテゴリ変数の処理と結合
        if x_cat is not None and len(self.embeddings) > 0:
            emb_outs = []
            for i, emb in enumerate(self.embeddings):
                # 各カテゴリ変数をembeddingし、[Batch, SeqLen, EmbDim]を得る
                emb_outs.append(emb(x_cat[:, :, i]))
            # 全ての入力を特徴量方向に結合
            x = torch.cat([x_cont] + emb_outs, dim=-1)
        else:
            x = x_cont
        # TCNの入力形式 [Batch, Channels, SeqLen] に変換
        x = x.permute(0, 2, 1)
        y = self.network(x)
        return self.fc(y[:, :, -1])