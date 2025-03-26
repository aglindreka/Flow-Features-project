import torch.nn as nn
from .Classification_Module import Classification_Module
from .TS_Mixer import Temporal_Mixer
from .Temporal_Encoder import TemporalEncoder


class MSTCT(nn.Module):
    """
    MS-TCT for action detection
    """
    def __init__(self, inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes):
        super(MSTCT, self).__init__()

        self.dropout=nn.Dropout()

        self.TemporalEncoder=TemporalEncoder(in_feat_dim=in_feat_dim, embed_dims=inter_channels,
                 num_head=head, mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,num_block=num_block)

        self.Temporal_Mixer=Temporal_Mixer(inter_channels=inter_channels, embedding_dim=final_embedding_dim)

        self.Classfication_Module=Classification_Module(num_classes=num_classes, embedding_dim=final_embedding_dim)
        self.los_destillation =  nn.L1Loss()
        self.linear_flow = nn.Linear(1568, 768)
        self.linear_depth = nn.Linear(128, 768)
        self.linear_pose = nn.Linear(256, 768)
        self.linear_SAM = nn.Linear(256, 768)
        self.linear_VLM = nn.Linear(512, 768)

    def forward(self, inputs, inputs_flow, inputs_depth, input_pose, input_SAM, input_VLM, is_train):
        if is_train:
            inputs = self.dropout(inputs)

            inputs_flow = self.linear_flow(inputs_flow.permute(0, 2, 1)).permute(0, 2, 1)
            inputs_depth = self.linear_depth(inputs_depth.permute(0, 2, 1)).permute(0, 2, 1)
            input_pose = self.linear_pose(input_pose.permute(0, 2, 1)).permute(0, 2, 1)
            input_SAM = self.linear_SAM(input_SAM.permute(0, 2, 1)).permute(0, 2, 1)
            input_VLM = self.linear_VLM(input_VLM.permute(0, 2, 1)).permute(0, 2, 1)

            los_destillation = self.los_destillation(inputs, inputs_flow) + self.los_destillation(inputs, inputs_depth)+ self.los_destillation(inputs, input_pose)+ self.los_destillation(inputs, input_SAM) + self.los_destillation(inputs, input_VLM)

            # Temporal Encoder Module
            x = self.TemporalEncoder(inputs)

            # Temporal Scale Mixer Module
            concat_feature, concat_feature_hm = self.Temporal_Mixer(x)

            # Classification Module
            x, x_hm = self.Classfication_Module(concat_feature, concat_feature_hm)
            return x, x_hm, los_destillation  # B, T, C
        else:
            inputs = self.dropout(inputs)

            # Temporal Encoder Module
            x = self.TemporalEncoder(inputs)

            # Temporal Scale Mixer Module
            concat_feature, concat_feature_hm = self.Temporal_Mixer(x)

            # Classification Module
            x, x_hm = self.Classfication_Module(concat_feature, concat_feature_hm)

            return x, x_hm, 0 # B, T, C





