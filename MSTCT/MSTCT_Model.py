import torch.nn as nn
from .Classification_Module import Classification_Module
from .TS_Mixer import Temporal_Mixer
from .Temporal_Encoder import TemporalEncoder
import torch
import numpy as np
import random
import os
from torch.nn import functional as F


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)
# torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class Expert_Regulator(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(Expert_Regulator, self).__init__()
        self.fc1_1 = nn.Conv1d(output_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.fc1_2 = nn.Conv1d(output_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.fc1_3 = nn.Conv1d(output_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.fc1_4 = nn.Conv1d(output_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.fc1_5 = nn.Conv1d(output_dim, hidden_dim, kernel_size=1, stride=1, padding=0)


        self.fc2_1 = nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)
        self.fc2_2 = nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)
        self.fc2_3 = nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)
        self.fc2_4 = nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)
        self.fc2_5 = nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)




    def forward(self, output1, output2, output3, output4, output5):



        hidden1 = F.relu(self.fc1_1(output1))
        hidden2 = F.relu(self.fc1_2(output2))
        hidden3 = F.relu(self.fc1_3(output3))
        hidden4 = F.relu(self.fc1_4(output4))
        hidden5 = F.relu(self.fc1_5(output5))



        expert1 = torch.sigmoid(self.fc2_1(hidden1))
        expert2 = torch.sigmoid(self.fc2_2(hidden2))
        expert3 = torch.sigmoid(self.fc2_3(hidden3))
        expert4 = torch.sigmoid(self.fc2_4(hidden4))
        expert5 = torch.sigmoid(self.fc2_5(hidden5))





        return expert1, expert2, expert3, expert4, expert5

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
        self.cross_attn1 = nn.MultiheadAttention(768, 8, batch_first=True)
        self.Expert_Regulator = Expert_Regulator(768, 32)

    def forward(self, inputs, inputs_flow, inputs_depth, input_pose, input_SAM, input_VLM, is_train):
        if is_train:
            inputs = self.dropout(inputs)

            inputs_flow = self.linear_flow(inputs_flow.permute(0, 2, 1))#.permute(0, 2, 1)
            inputs_depth = self.linear_depth(inputs_depth.permute(0, 2, 1))#.permute(0, 2, 1)
            input_pose = self.linear_pose(input_pose.permute(0, 2, 1))#.permute(0, 2, 1)
            input_SAM = self.linear_SAM(input_SAM.permute(0, 2, 1))#.permute(0, 2, 1)
            input_VLM = self.linear_VLM(input_VLM.permute(0, 2, 1))#.permute(0, 2, 1)
            flow_expert, depth_expert, pose_expert, SAM_expert, VLM_expert = self.Expert_Regulator(inputs_flow.permute(0, 2, 1), inputs_depth.permute(0, 2, 1), input_pose.permute(0, 2, 1), input_SAM.permute(0, 2, 1), input_VLM.permute(0, 2, 1))
            all = torch.cat((flow_expert.permute(0, 2, 1)*inputs_flow, depth_expert.permute(0, 2, 1)*inputs_depth, pose_expert.permute(0, 2, 1)*input_pose, SAM_expert.permute(0, 2, 1)*input_SAM, VLM_expert.permute(0, 2, 1)*input_VLM), dim=1)

            all, _  = self.cross_attn1(inputs.permute(0, 2, 1), all, all)

            los_destillation = self.los_destillation(inputs, all.permute(0, 2, 1))# + self.los_destillation(inputs, inputs_depth)+ self.los_destillation(inputs, input_pose)+ self.los_destillation(inputs, input_SAM) + self.los_destillation(inputs, input_VLM)

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





