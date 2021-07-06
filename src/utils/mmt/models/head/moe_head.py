import torch.nn as nn

from src.utils.mmt.models import HEAD, build_loss


@HEAD.register_module()
class MoEHead(nn.Module):
    def __init__(self, in_dim, out_dim, loss, dropout_p=None, num_experts=4):
        super(MoEHead, self).__init__()
        self.gate_fc = nn.Linear(in_dim, out_dim*(num_experts+1))
        self.expert_fc = nn.Linear(in_dim, out_dim*num_experts)
        if dropout_p is None:
            dropout_p = 1e-11
        self.dropout = nn.Dropout(dropout_p)
        self.loss = build_loss(loss)
        self.num_experts = num_experts
        self.num_classes = out_dim

    def forward_train(self, x, gt_labels):
        x = self.dropout(x)
        gate_act = self.gate_fc(x).reshape((-1, self.num_experts+1))
        expert_act = self.expert_fc(x).reshape((-1, self.num_experts))
        gate_attn = gate_act.softmax(1)[:, :-1]
        expert_pred = expert_act.sigmoid()
        expert_pred = (gate_attn * expert_pred).mean(1)
        pred = expert_pred.reshape((-1, self.num_classes))
        loss_list = [self.loss(pred[i], gt_labels[i]) for i in range(len(x))]
        return dict(cls_loss=loss_list)

    def simple_test(self, x):
        gate_act = self.gate_fc(x).reshape((-1, self.num_experts+1))
        expert_act = self.expert_fc(x).reshape((-1, self.num_experts))
        gate_attn = gate_act.softmax(1)[:, :-1]
        expert_pred = expert_act.sigmoid()
        expert_pred = (gate_attn * expert_pred).mean(1)
        pred = expert_pred.reshape((-1, self.num_classes))
        return pred

