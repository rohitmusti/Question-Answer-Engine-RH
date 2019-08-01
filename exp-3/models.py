import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class classifier(nn.Module):
    def __init__(self, args, word_vectors):
        super(classifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings=word_vectors)
        # self.lstm = nn.LSTM(input_size=args.ques_limit, hidden_size=args.hidden_size,
        #                     num_layers=args.LSTM_num_layers, bias=True,
        #                     dropout=0.2)

        # 300 size input because that is the size of each individual word vector
        self.lstm = nn.LSTM(input_size=300, hidden_size=args.hidden_size, 
                            num_layers=args.LSTM_num_layers, bias=True, 
                            dropout = (args.LSTM_dropout if args.LSTM_num_layers > 1 else 0) ) 
        self.full_1 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.full_2 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.conv = nn.Conv1d(in_channels=1, 
                              out_channels=1,
                              kernel_size=5) # kernel size is prob 3 I think I'm approaching this incorrectly
        self.pool = nn.MaxPool1d(kernel_size=2) # not really sure waht the kernel size is
        self.full_3 = nn.Linear(in_features=498, out_features=1)
        self.full_4 = nn.Linear(in_features=1, out_features=442)
        self.out = nn.Sigmoid()

    def forward(self, qw_idxs, lengths):
        
        qw_vec = self.embedding(qw_idxs)

        # batch_size, w_vec_len, num_word = qw_vec.size()

        # print(f"batch_size: {batch_size}")
        # print(f"number of words in sentence: {w_vec_len}")
        # print(f"word vector length: {num_word}")
        # print(f"lengths dim: {lengths.size()}")
        # print(f"lengths: {lengths}")
        # print(f"qw_vec size: {qw_vec.size()}")

        # here I need to sort the elements in each batch by length 
        # and then do a pack padded sequence on them

        # orig_len = qw_vec.size(1)
        lengths, sort_index = lengths.sort(0, descending=True)
        qw_vec = Variable(qw_vec[sort_index])
        qw_vec = pack_padded_sequence(input=qw_vec, lengths=lengths, batch_first=True).float()

        # the output of the lstm is descripted as the output of lstm, the hidden state of 
        # the last time step and the cell state for the last time step
        # the hidden state is the best representation of the information I think
        # might be worth setting up another experiment that uses the actual last output

        lstm_out, (h_n, c_n) = self.lstm(qw_vec)

        # it turns that b/c I was using the hidden state I didn't need this, but
        # I am keeping it just in case it becomes necessary

#        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=orig_len)
#        _, unsort_idx = sort_index.sort(0)
#        lstm_out = lstm_out[unsort_idx]

        h_n = h_n.permute(1,0,2).squeeze()
 
        f1_out = self.full_1(h_n)
        f2_out = self.full_2(f1_out)
        f2_out = torch.unsqueeze(f2_out, dim=1)

        c_out = self.conv(f2_out)
        p_out = self.pool(c_out)

        f3_out = self.full_3(p_out)
        f4_out = self.full_4(f3_out)
        f4_out = torch.squeeze(f4_out)

        out = self.out(f4_out).float()

        return out


