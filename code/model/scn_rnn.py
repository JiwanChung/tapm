import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SCNLSTM(nn.Module):
    def __init__(self, input_size, tag_size, hidden_size, n_layers, batch_first, dropout):
        super(SCNLSTM, self).__init__()
        nh = hidden_size #512
        nf = hidden_size #512
        nk = tag_size #300
        nx = input_size
        self.num_layers = n_layers
        self.out_dim = nh
        self._all_weights = []
        for layer in range(n_layers):
            if layer > 0:
                nx = nh
            for suffix in ['i', 'a', 'o', 'c']:
                w_a = Parameter(torch.Tensor(nf, nh))
                w_b = Parameter(torch.Tensor(nk, nf))
                w_c = Parameter(torch.Tensor(nx, nf))
                u_a = Parameter(torch.Tensor(nf, nh))
                u_b = Parameter(torch.Tensor(nk, nf))
                u_c = Parameter(torch.Tensor(nh, nf))
                b = Parameter(torch.Tensor(1, nh))
                layer_params = (w_a, w_b, w_c, u_a, u_b, u_c, b)
                # initialize
                for l in layer_params[:-1]:
                    torch.nn.init.xavier_uniform_(l)
                torch.nn.init.constant_(b, 0)
                param_names = ['weight_wa_{}{}', 'weight_wb_{}{}', 'weight_wc_{}{}', 'weight_ua_{}{}', 'weight_ub_{}{}', 'weight_uc_{}{}', 'bias_{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

    def init_c(self, B, C, device=0):
        return torch.zeros(B, self.num_layers, C).float().to(device)

    def init_h(self, B, device=0):
        h = (torch.zeros(B, self.num_layers, self.out_dim).float().to(device),
                torch.zeros(B, self.num_layers, self.out_dim).float().to(device))
        return h

    def forward(self, inps, hidden, keyword):
        '''
          inps: B x T x nx
          tag: B x nk
          hns, cns: N x B x nh (N: rnn_layers)
        return
          output: B x T x nh
        '''
        tag = keyword.float()
        hns, cns = hidden
        hns = hns.transpose(0, 1)
        cns = cns.transpose(0, 1)
        hn_list, cn_list = [], []
        for layer in range(self.num_layers):
            hn_list.append(hns[layer, :, :])
            cn_list.append(cns[layer, :, :])
        T = inps.shape[1]
        output = []
        for i in range(T):
            inp = inps[:,i,:]
            for layer in range(self.num_layers):
                hn = hn_list[layer]
                cn = cn_list[layer]
                suff_res = []
                for suffix in ['i', 'a', 'o', 'c']:
                    # B x nf
                    tmpx1 = torch.matmul(tag, getattr(self, 'weight_wb_{}{}'.format(layer, suffix)))
                    # B x nf
                    tmpx2 = torch.matmul(inp, getattr(self, 'weight_wc_{}{}'.format(layer, suffix)))
                    # B x nf
                    tmph1 = torch.matmul(tag, getattr(self, 'weight_ub_{}{}'.format(layer, suffix)))
                    # B x nf
                    tmph2 = torch.matmul(hn, getattr(self, 'weight_uc_{}{}'.format(layer, suffix)))
                    # B x nf
                    tmpx = tmpx1 * tmpx2
                    # B x nf
                    tmph = tmph1 * tmph2
                    # B x nh
                    tmpx3 = torch.matmul(tmpx, getattr(self, 'weight_wa_{}{}'.format(layer, suffix)))
                    # B x nh
                    tmph3 = torch.matmul(tmph, getattr(self, 'weight_ua_{}{}'.format(layer, suffix)))
                    # B x nh
                    tmp = torch.sigmoid(tmpx3 + tmph3 + getattr(self, 'bias_{}{}'.format(layer, suffix)))
                    suff_res.append(tmp)
                cn = suff_res[0] * suff_res[3] + suff_res[1] * cn
                hn = suff_res[2] * torch.tanh(cn)
                # Update
                inp = hn
                hn_list[layer] = hn
                cn_list[layer] = cn
            output.append(hn)
        output = torch.stack(output, 1) # B x T x nh
        hns = torch.stack(hn_list, 0) # N x B x nh
        cns = torch.stack(cn_list, 0) # N x B x nh
        hns = hns.transpose(0, 1)
        cns = cns.transpose(0, 1)
        return output, (hns, cns)
