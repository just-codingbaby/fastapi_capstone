import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from app.db_process import connect_to_mysql

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random_seed = 1

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.set_printoptions(precision = 8)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Mlp_feat(nn.Module):
    def __init__(self, in_features, hidden_dim, hidden_features=None, out_features=None, drop=0.):
        super(Mlp_feat, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_dim or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x): # B, L, D -> B, L, D
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_time(nn.Module):
    def __init__(self, in_features, hidden_dim, hidden_features=None, out_features=None, drop=0.):
        super(Mlp_time, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_dim or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features) # 수정함

    def forward(self, x): # B, D, L -> B, D, L
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # 수정함
        return x

class Mixer_Layer(nn.Module):
    def __init__(self, time_dim, feat_dim, hidden_dim):
        super(Mixer_Layer, self).__init__()

        self.batchNorm2D = nn.BatchNorm1d(time_dim)
        self.MLP_time = Mlp_time(time_dim, hidden_dim)
        self.MLP_feat = Mlp_feat(feat_dim, hidden_dim)

    def forward(self, x):
        res1 = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res1

        res2 = x
        x = self.batchNorm2D(x)
        x = self.MLP_feat(x)
        x = x + res2
        return x

class Backbone(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, hidden_dim, layer_num = 1):
        super(Backbone, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.layer_num = layer_num = 1

        self.mix_layer = Mixer_Layer(seq_len, enc_in, hidden_dim)
        self.temp_proj = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = self.mix_layer(x)
        x = self.temp_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class TSMixer(nn.Module):

    def __init__(self, enc_in, seq_len, pred_len, hidden_dim):
        super(TSMixer, self).__init__()
        self.rev = RevIN(enc_in)
        self.backbone = Backbone(seq_len, pred_len, enc_in, hidden_dim)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x):
        z = self.rev(x, 'norm')
        z = self.backbone(z)
        z = self.rev(z, 'denorm')
        return z

# enc_in = len(df.columns)
# seq_len = 24
# pred_len = 1
# patch_len = 1
# dropout = 0.1
# hidden_dim = 128
#
# example = model = TSMixer(enc_in, seq_len, pred_len, 32).to(device)
# example.load_state_dict(torch.load('example.pkl'))
#
# model_test = df.iloc[24:48].to_numpy() #프론트로부터 입력받은 시간에 따라 이 코드가 변경됨
# model_test_scaled = torch.FloatTensor(model_test).unsqueeze(0).to(device)
#
# example.eval()
#
# prediction = []
# with torch.no_grad() :
#     prediction = example(model_test_scaled)
#
# pred = prediction[:,-1, :].cpu().numpy()
# result = pred[0][-1]
# print(result)


def create_dataframe_from_table(table_name, line):  # table -> df 전환
    connection = connect_to_mysql(line)
    cursor = connection.cursor()
    query = f"SELECT * FROM `{table_name}`"
    cursor.execute(query)
    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].apply(lambda row: row.year, 1)
    df['month'] = df['datetime'].apply(lambda row: row.month, 1)
    df['date'] = df['datetime'].apply(lambda row: row.day, 1)
    df['hour'] = df['datetime'].apply(lambda row: row.hour, 1)
    df['minute'] = df['datetime'].apply(lambda row: row.minute, 1)
    df['weekday'] = df['datetime'].apply(lambda row: row.weekday(), 1)

    df.set_index('datetime', drop=True, inplace=True)
    df = df[['TRFFCVLM', 'year', 'month', 'date', 'hour', 'minute', 'weekday', 'SPD_AVG']]
    cursor.close()
    return df