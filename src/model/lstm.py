import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_hidden, H, W)
        conv_out = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels[i - 1]
            self.layers.append(ConvLSTMCell(in_channels, hidden_channels[i], kernel_size))

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        h, c = [], []

        for i in range(self.num_layers):
            h.append(torch.zeros(B, self.hidden_channels[i], H, W, device=x.device))
            c.append(torch.zeros(B, self.hidden_channels[i], H, W, device=x.device))

        for t in range(T):
            input_t = x[:, t]  # shape: (B, C, H, W)
            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(input_t, h[i], c[i])
                input_t = h[i]

        return h[-1]  # Final hidden state of last layer


class ConvLSTMForecast(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        hidden_channels=[32, 64],
        kernel_size=3,
        num_layers=2,
        output_vars=['tas', 'pr']
    ):
        super().__init__()
        assert n_output_channels == len(output_vars), "Mismatch between output channels and output_vars list"

        self.output_vars = output_vars
        self.convlstm = ConvLSTM(n_input_channels, hidden_channels, kernel_size, num_layers)

        # Separate output heads for each variable
        self.heads = nn.ModuleDict()
        for var in output_vars:
            self.heads[var] = nn.Sequential(
                nn.Conv2d(hidden_channels[-1], 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=1)  # Output 1 channel per variable
            )

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Handle single-frame input
    
        features = self.convlstm(x)  # (B, hidden, H, W)
        outs = [self.heads[var](features) for var in self.output_vars]
        out = torch.cat(outs, dim=1)  # (B, C_out, H, W)
        return out