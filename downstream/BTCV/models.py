import torch
from monai.networks.nets import SwinUNETR


class SwinUNETRDS(SwinUNETR):
    """SwinUNETR with deep-supervision auxiliary heads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fs = kwargs["feature_size"]
        oc = kwargs["out_channels"]
        chans = (fs, 2 * fs, 4 * fs, 8 * fs)
        self.ds_heads = torch.nn.ModuleList([torch.nn.Conv3d(c, oc, kernel_size=1) for c in chans])

    def forward(self, x_in):
        hs = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hs[0])
        enc2 = self.encoder3(hs[1])
        enc3 = self.encoder4(hs[2])
        dec4 = self.encoder10(hs[4])
        dec3 = self.decoder5(dec4, hs[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        if self.training:
            feats = [dec0, dec1, dec2, dec3]
            return [logits] + [head(f) for head, f in zip(self.ds_heads, feats)]
        return logits
