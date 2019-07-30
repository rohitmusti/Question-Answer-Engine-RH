import torch
import torch.nn as nn

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        embedding = nn.Embedding.from_pretrained(embeddings=word_vectors)
        lstm = nn.LSTM(input_size=args.ques_limit, hidden_size=args.LSTM_hidden_size,
                       num_layers=args.LSTM_num_layers, bias=True,
                       dropout=0.2)
        full_1 = nn.Linear(in_features=args.LSTM_hidden_size, out_features=args.LSTM_hidden_size)
        full_2 = nn.Linear(in_features=args.LSTM_hidden_size, out_features=args.LSTM_hidden_size)
        conv = nn.conv1d(in_channels=, 
                         out_channels=,
                         kernel_size=)
        full_3 = nn.Linear(in_features=args.LSTM_hidden_size, out_features=args.LSTM_hidden_size)
        full_4 = nn.Linear(in_features=args.LSTM_hidden_size, out_features=args.LSTM_hidden_size)
        out = None
    
        model = nn.Sequential(embedding, 
                              lstm, 
                              full_1, 
                              full_2, 
                              conv, 
                              pool, 
                              full_3, 
                              full_4, 
                              out)