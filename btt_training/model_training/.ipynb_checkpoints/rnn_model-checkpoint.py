import torch 
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # # Parameters for the day-specific input layers
        # # self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # self.day_layer_activation = nn.Softsign()
        # day_weights_init = torch.stack(
        #     [torch.eye(neural_dim) for _ in range(n_days)],
        #     dim=0,  # [n_days, D, D]
        # )
        # self.day_weights = nn.Parameter(day_weights_init)  # [n_days, D, D]
        # self.day_biases  = nn.Parameter(torch.zeros(n_days, neural_dim))  # [n_days, D]
        
        # # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        # self.day_weights = nn.ParameterList(
        #     [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        # )
        # self.day_biases = nn.ParameterList(
        #     [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        # )

        # self.day_layer_dropout = nn.Dropout(input_dropout)
        
        # self.input_size = self.neural_dim
        
        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign()  # basically a shallower tanh 

        # Use single tensors instead of ParameterList for torch.compile compatibility
        day_weights_init = torch.stack(
            [torch.eye(self.neural_dim) for _ in range(self.n_days)],
            dim=0,  # [n_days, D, D]
        )
        self.day_weights = nn.Parameter(day_weights_init)  # [n_days, neural_dim, neural_dim]

        day_biases_init = torch.zeros(self.n_days, self.neural_dim)  # [n_days, D]
        self.day_biases = nn.Parameter(day_biases_init)

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim


        
        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, lengths=None, states=None, return_state=False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        # day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        # day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        # x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        # x = self.day_layer_activation(x)
        # x: [B, T, D], day_idx: [B]
        day_w = self.day_weights[day_idx]                  # [B, D, D]
        day_b = self.day_biases[day_idx].unsqueeze(1)      # [B, 1, D]

        x = torch.matmul(x, day_w) + day_b
        
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pack sequences to skip padded timesteps
        # if lengths is not None:
        #     # lengths는 패치 전 시간축 길이로 들어올 수 있으므로, 패치 사용 시 패치 후 길이로 변환
        #     if self.patch_size > 0:
        #         # 패치 후 길이 P = floor((L - K) / S) + 1
        #         patched_lengths = ((lengths - self.patch_size) // self.patch_stride) + 1
        #         patched_lengths = torch.clamp(patched_lengths, min=1)
        #         eff_lengths = patched_lengths
        #     else:
        #         eff_lengths = lengths

        #     eff_lengths = eff_lengths.to(x.device).to(torch.int64).detach().cpu()
        #     packed = pack_padded_sequence(x, eff_lengths, batch_first=True, enforce_sorted=False)
        #     packed_output, hidden_states = self.gru(packed, states)
        #     output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # else:
        #     # 길이가 없으면 기존처럼 전체 타임스텝 연산
        #     output, hidden_states = self.gru(x, states)

        # GRU, 패딩 구간을 건너뛰도록 패킹
        if lengths is not None:
            if self.patch_size > 0:
                # 패치 사용 시, 길이를 패치 후 길이로 변환
                eff_lengths = ((lengths - self.patch_size) // self.patch_stride) + 1
                eff_lengths = torch.clamp(eff_lengths, min=1)
            else:
                eff_lengths = lengths
            packed = pack_padded_sequence(
                x, eff_lengths.to(torch.int64).detach().cpu(),
                batch_first=True, enforce_sorted=False
            )
            packed_out, hidden_states = self.gru(packed, states)
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits
        

