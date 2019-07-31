import torch
import torch.nn as nn

class classifier(nn.Module):
    def __init__(self, args, word_vectors):
        super(classifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings=word_vectors)
        # self.lstm = nn.LSTM(input_size=args.ques_limit, hidden_size=args.hidden_size,
        #                     num_layers=args.LSTM_num_layers, bias=True,
        #                     dropout=0.2)
        self.lstm = nn.LSTM(31, args.hidden_size, num_layers=2)
        self.full_1 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.full_2 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.conv = nn.Conv1d(in_channels=args.hidden_size, 
                              out_channels=args.hidden_size,
                              kernel_size=3) # kernel size is prob 3
        self.pool = nn.MaxPool1d(kernel_size=3) # not really sure waht the kernel size is
        self.full_3 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.full_4 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.out = nn.Sigmoid()

    def forward(self, qw_idx):
        
        qw_vec = self.embedding(qw_idx)
        batch_size, w_vec_len, num_word = qw_vec.size()
        # print(f"batch_size: {batch_size}")
        # print(f"number of words in sentence: {w_vec_len}")
        # print(f"word vector length: {num_word}")
        qw_vec = qw_vec.permute(0, 2, 1)
        lstm_out = self.lstm(qw_vec)
        f1_out = self.full_1(lstm_out)
        f2_out = self.full_2(f1_out)
        c_out = self.conv(f2_out)
        p_out = self.pool(c_out)
        f3_out = self.full_3(p_out)
        f4_out = self.full_4(p_out)
        out = self.out(f4_out)

        return lstm_out


