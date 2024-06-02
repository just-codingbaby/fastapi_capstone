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

class SegRnn(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, patch_len, dropout, hidden_dim):
        super(SegRnn, self).__init__()

        self.lucky = nn.Embedding(enc_in, hidden_dim // 2)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim

        self.linear_patch = nn.Linear(self.patch_len, self.hidden_dim)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.hidden_dim // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.hidden_dim // 2))

        self.dropout = nn.Dropout(dropout)
        self.linear_patch_re = nn.Linear(self.hidden_dim, self.patch_len)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.hidden_dim

        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        enc_out = self.gru(enc_in)[1].repeat(1, 1, M).view(1, -1, self.hidden_dim)  # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1),  # M, d // 2 -> 1, M, d // 2 -> B * C, M, d // 2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1)  # C, d // 2 -> C, 1, d // 2 -> B * C, M, d // 2
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        dec_out = self.gru(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1)  # B, C, H

        y = y + seq_last
        return y

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