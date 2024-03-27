
 
import torch.nn as nn
from architectures.UNET.attention_block import AttentionBlock  
 
from architectures.UNET.normal_attention import NormalAttentionBlock
from architectures.UNET.resblock import ResBlock 
 
class ConvBlock(nn.Module):
    

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        dropout=0.02, 
        groupnorm=16,
        num_head_channels=-1,
        attention_resolution = [4,8,16],
        num_heads=8,
        use_scale_shift_norm=False,
         resolution=1,
        up=False,
        down=False,
        use_conv=True,
        use_conv_up_down=False,
        use_new_attention_order=False,
        use_flash_attention=False):
        super().__init__() 
        self.res= resolution
        self.self_att=None
        self.resblock= ResBlock(in_channels=in_channels,out_channels=out_channels,dropout=dropout,emb_channels=emb_channels, 
                                use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm, up=up,down=down, use_conv=use_conv, use_conv_up_down=use_conv_up_down)
        if resolution in attention_resolution: 
            if use_flash_attention:
                self.self_att=  AttentionBlock(channels=out_channels,num_heads=num_heads,groupnorm_ch=groupnorm, num_head_channels=num_head_channels)  
            else:        
                self.self_att=  NormalAttentionBlock(channels=out_channels,num_heads=num_heads,groupnorm_ch=groupnorm,    num_head_channels=num_head_channels,  use_new_attention_order=use_new_attention_order)  
    
            print(resolution)
    def forward(self, x, emb):
        x = self.resblock(x,emb)
        if self.self_att: 
            x= self.self_att(x)
        return x


 