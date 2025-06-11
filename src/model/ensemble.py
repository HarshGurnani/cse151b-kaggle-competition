import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, convlstm, transformer, cnn, output_vars=['tas', 'pr'], weights_tas=None, weights_pr=None):
        super().__init__()
        self.convlstm = convlstm
        self.transformer = transformer
        self.cnn = cnn
        self.output_vars = output_vars

        # Default to equal weights if not provided
        self.weights_tas = weights_tas or [1.0, 1.0, 1.0]
        self.weights_pr = weights_pr or [1.0, 1.0, 1.0]

        assert len(self.weights_tas) == 3 and len(self.weights_pr) == 3, "Each weight list must have 3 values"

    def forward(self, x_seq, x_img=None):
        """
        Inputs:
            x_seq: (B, T, C, H, W)
            x_img: optional (B, C, H, W); if None, defaults to last frame in x_seq
        Output:
            (B, C_out, H, W)
        """

        # If x_img not given, derive from x_seq
        if x_img is None:
            if x_seq.dim() == 4:
                x_seq = x_seq.unsqueeze(1)  # Make it (B, 1, C, H, W)
            if x_seq.dim() == 5:
                x_img = x_seq[:, -1]
            else:
                raise ValueError(f"x_seq must be 4D or 5D, got shape: {x_seq.shape}")

        # Ensure x_img is 4D
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(0)
        elif x_img.dim() == 5:
            x_img = x_img[:, -1]
        elif x_img.dim() != 4:
            raise ValueError(f"x_img must be 4D, got shape: {x_img.shape}")

        # Get outputs
        # out_lstm = self.convlstm(x_seq)       # (B, 2, H, W)
        out_trans = self.transformer(x_img)   # (B, 2, H, W)
        # out_cnn = self.cnn(x_img)             # (B, 2, H, W)

        # Split channels
        # tas = (
        #     self.weights_tas[0] * out_lstm[:, 0] +
        #     self.weights_tas[1] * out_trans[:, 0] +
        #     self.weights_tas[2] * out_cnn[:, 0]
        # ) / sum(self.weights_tas)

        # pr = (
        #     self.weights_pr[0] * out_lstm[:, 1] +
        #     self.weights_pr[1] * out_trans[:, 1] +
        #     self.weights_pr[2] * out_cnn[:, 1]
        # ) / sum(self.weights_pr)

        tas = self.weights_tas[1] * out_trans[:, 0]
        pr = self.weights_pr[1] * out_trans[:, 1]

        # Stack back to (B, 2, H, W)
        ensemble_out = torch.stack([tas, pr], dim=1)
        return ensemble_out
