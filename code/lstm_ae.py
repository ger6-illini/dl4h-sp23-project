import os
import numpy as np
import torch
import torch.nn as nn

#----------------------------------------------------------------
# Implementation of Early Stopping to be used in LSTM Autoencoder
# source : https://github.com/Bjarten/early-stopping-pytorch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=20, verbose=False, delta=-0.00001):
        """
        Parameters
        ----------
        patience : int, default 7
            How long to wait after last time validation loss improved.
        verbose : bool, default False
            If True, prints a message for each validation loss improvement.
        delta : float, default 0
            Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f"Early Stopping activated. Final validation loss : {self.val_loss_min:.7f}")
                self.early_stop = True
        # if the current score does not exceed the best scoee, run the following code below
        else:  
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), './checkpoint.pt')
        self.val_loss_min = val_loss

#-----------------
# LSTM Autoencoder
# code inspired by https://github.com/hellojinwoo/TorchCoder/blob/master/autoencoders/rae.py that
# was inspired by https://github.com/shobrook/sequitur/blob/master/sequitur/autoencoders/rae.py
# annotation sourced by https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM

# (1) Encoder
class Encoder(nn.Module):
    def __init__(self, seq_len, num_features, embedding_size):
        super().__init__()

        self.seq_len = seq_len
        self.num_features = num_features         # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size     # the size of the resulting embedding (LSTM hidden states)
        self.LSTM = nn.LSTM(
            input_size = num_features,
            hidden_size = embedding_size,
            num_layers = 1,
            batch_first = True
        )

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (hidden_state, cell_state) = self.LSTM(x)
        last_lstm_layer_hidden_state = hidden_state[-1, :, :]
        return last_lstm_layer_hidden_state

# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, num_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.num_features = num_features
        self.output_size = output_size
        self.LSTM = nn.LSTM(
            input_size = num_features,
            hidden_size = self.output_size,
            num_layers = 1,
            batch_first = True
        )

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM(x)
        x = x.reshape((-1, self.seq_len, self.output_size))
        return x

# (3) Autoencoder: putting the encoder and decoder together
class LSTM_AE(nn.Module):
    def __init__(
        self, seq_len, num_features, embedding_dim,
        learning_rate=1e-4,
        every_epoch_print=False,
        epochs=100,
        patience=10,
        max_grad_norm=1):

        super().__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(self.seq_len, self.num_features, self.embedding_dim)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.num_features)
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.every_epoch_print = every_epoch_print
    
    def forward(self, x):
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def fit(self, x):
        """
        Trains the model's parameters over a fixed number of epochs, specified by `epochs`, as long as the loss keeps decreasing.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        criterion = nn.MSELoss(reduction='mean')
        self.train()
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=False)

        for epoch in range(1 , self.epochs + 1):
            # updating early_stopping's epoch
            early_stopping.epoch = epoch
            optimizer.zero_grad()
            encoded, decoded = self(x)
            loss = criterion(decoded, x)

            # early_stopping needs the validation loss to check if it has decresed, 
            # and, if it has, it will make a checkpoint of the current model
            early_stopping(loss, self)
            
            if early_stopping.early_stop:
                break

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)
            optimizer.step()

            if epoch % self.every_epoch_print == 0:
                print(f"epoch : {epoch}, loss_mean : {loss.item():.7f}")

        # load the last checkpoint with the best model
        self.load_state_dict(torch.load('./checkpoint.pt'))

        # to check the final_loss
        encoded, decoded = self(x)
        final_loss = criterion(decoded , x).item()
        
        return final_loss
    
    def encode(self, x):
        self.eval()
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, x):
        self.eval()
        decoded = self.decoder(x)
        squeezed_decoded = decoded.squeeze()
        return squeezed_decoded
    
    def load(self, PATH):
        """
        Loads the model's parameters from path `PATH`.
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))
