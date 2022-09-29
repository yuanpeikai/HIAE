import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from helper import *


class InteractE(torch.nn.Module):
    def __init__(self, params):
        super(InteractE, self).__init__()

        self.p = params

        self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim)  # # batch_size,400
        xavier_normal_(self.ent_embed.weight)
        self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2 + 1, self.p.embed_dim)
        xavier_normal_(self.rel_embed.weight)

        if self.p.agg_func == 'HyperCrossE':
            self.ent_embed_w = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim)  # # batch_size,400
            xavier_normal_(self.ent_embed_w.weight)
            self.rel_embed_w = torch.nn.Embedding(self.p.num_rel * 2 + 1, self.p.embed_dim)
            xavier_normal_(self.rel_embed_w.weight)

        self.bceloss = torch.nn.BCELoss()

        self.inp_drop = torch.nn.Dropout(self.p.inp_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.feat_drop)
        self.aware_drop = torch.nn.Dropout(0.1)

        self.chequer_perm = self.get_chequer_perm()

        self.w_sub = get_param((self.p.embed_dim * 2, 1))
        self.w_rel = get_param((self.p.embed_dim * 2, 1))

        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        self.bn0 = torch.nn.BatchNorm2d(self.p.iperm)
        self.bn1 = torch.nn.BatchNorm2d(self.p.channel * self.p.iperm)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))  # num_entities
        self.register_parameter('conv_filt',
                                Parameter(torch.zeros(self.p.channel, 1, self.p.filter_size, self.p.filter_size)))
        xavier_normal_(self.conv_filt) - 1

        self.flat_sz = self.p.ik_h * 2 * self.p.ik_w * self.p.channel * self.p.iperm
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

        if self.p.method == 'perceptual':
            self.mlp = torch.nn.Linear(self.p.embed_dim * 2, self.p.embed_dim)
        elif self.p.method == 'co-aware':
            self.w_method = get_param((self.p.embed_dim, 1))
            self.w_sub_aware = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
            self.w_rel_aware = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)

        print('InteractE')

    def loss(self, pred, true_label=None):
        loss = self.bceloss(pred, true_label)
        return loss

    def agg(self, sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N, sub_emb_enlarge=0, rel_emb_enlarge=0):

        if self.p.agg_func == 'HyperCrossE':
            quals_ent_emb, quals_rel_emb = HyperCrossE(sub_emb_enlarge, rel_emb_enlarge, quals_ent_emb, quals_rel_emb)

        sub_score = self.att(sub_emb, quals_rel_emb, N, True)
        rel_score = self.att(rel_emb, quals_rel_emb, N, False)

        emb = self.common(sub_score, rel_score, quals_ent_emb, self.p.method)

        sub = sub_emb * self.p.alpha + (1 - self.p.alpha) * emb
        rel = rel_emb * self.p.alpha + (1 - self.p.alpha) * emb

        return sub, rel

    def att(self, triple_emb, score_quals, N, is_sub):
        triple_emb = triple_emb.unsqueeze(1).repeat(1, score_quals.size(1), 1)
        all_message = torch.cat([triple_emb, score_quals], dim=-1)

        if (is_sub):
            score = torch.matmul(all_message, self.w_sub).squeeze(-1)  # 128,4
        else:
            score = torch.matmul(all_message, self.w_rel).squeeze(-1)

        score = -self.leakyrelu(score)
        num_inf = torch.full_like(score, -np.inf)
        score = torch.where(N.byte(), score, num_inf)

        score = torch.exp(score)
        score_all = score.sum(dim=-1)
        score_all[score_all == 0.0] = 1.0
        score = score.div(score_all.unsqueeze(-1))
        return score

    def common(self, sub_score, rel_score, quals_ent_emb, fusion='co-aware'):

        sub_score = sub_score.unsqueeze(-1).repeat(1, 1, self.p.embed_dim)
        sub_aware_emb = quals_ent_emb * sub_score
        sub_aware_emb = sub_aware_emb.sum(dim=1)

        rel_score = rel_score.unsqueeze(-1).repeat(1, 1, self.p.embed_dim)
        rel_aware_emb = quals_ent_emb * rel_score
        rel_aware_emb = rel_aware_emb.sum(dim=1)

        if fusion == 'mult':
            return sub_aware_emb * rel_aware_emb
        elif fusion == 'perceptual':
            score_emb = torch.cat([sub_aware_emb, rel_aware_emb], dim=-1)
            return self.mlp(score_emb)
        elif fusion == 'co-aware':
            sub_emb_score = torch.matmul(sub_aware_emb, self.w_method).squeeze(-1)
            rel_emb_score = torch.matmul(rel_aware_emb, self.w_method).squeeze(-1)
            sub_emb_score = -self.leakyrelu(sub_emb_score)
            rel_emb_score = -self.leakyrelu(rel_emb_score)

            sub_emb_score = torch.exp(sub_emb_score)
            rel_emb_score = torch.exp(rel_emb_score)
            score_all = sub_emb_score + rel_emb_score
            sub_emb_score = (sub_emb_score / score_all).unsqueeze(-1).repeat(1, self.p.embed_dim)
            rel_emb_score = (rel_emb_score / score_all).unsqueeze(-1).repeat(1, self.p.embed_dim)
            return sub_emb_score * sub_aware_emb + rel_emb_score * rel_aware_emb

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])

        comb_idx = []
        for k in range(self.p.iperm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.ik_h):
                for j in range(self.p.ik_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        # chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        chequer_perm = torch.LongTensor(np.int32(comb_idx))
        return chequer_perm

    def forward(self, sub, rel, quals, N):
        self.ent_embed.weight.data[0] = 0
        self.rel_embed.weight.data[0] = 0

        if self.p.agg_func == 'HyperCrossE':
            self.ent_embed_w.weight.data[0] = 0
            self.rel_embed_w.weight.data[0] = 0

        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)  # batch_size,100

        quals_ent_emb = self.ent_embed(quals[:, 1::2])
        quals_rel_emb = self.rel_embed(quals[:, 0::2])

        if self.p.agg_func == 'HyperCrossE':
            sub_emb_enlarge = self.ent_embed_w(sub).unsqueeze(1).repeat(1, quals_rel_emb.size(1), 1)
            rel_emb_enlarge = self.rel_embed_w(rel).unsqueeze(1).repeat(1, quals_rel_emb.size(1), 1)
            sub_emb, rel_emb = self.agg(sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N, sub_emb_enlarge,
                                        rel_emb_enlarge)
        else:
            sub_emb, rel_emb = self.agg(sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N)

        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.iperm, 2 * self.p.ik_w, self.p.ik_h))
        stack_inp = self.bn0(stack_inp)
        x = self.inp_drop(stack_inp)
        x = self.circular_padding_chw(x, self.p.filter_size // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.iperm, 1, 1, 1), padding=0, groups=self.p.iperm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))  # batch_size,num_entities
        x += self.bias.expand_as(x)  # batch_size,num_entities

        pred = torch.sigmoid(x)  # batch_size,num_entities

        return pred


class Transformer_Triple(torch.nn.Module):
    def __init__(self, params):
        super(Transformer_Triple, self).__init__()

        self.p = params

        self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim)  # # batch_size,400
        xavier_normal_(self.ent_embed.weight)
        self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2 + 1, self.p.embed_dim)
        xavier_normal_(self.rel_embed.weight)

        if self.p.agg_func == 'HyperCrossE':
            self.ent_embed_w = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim)  # # batch_size,400
            xavier_normal_(self.ent_embed_w.weight)
            self.rel_embed_w = torch.nn.Embedding(self.p.num_rel * 2 + 1, self.p.embed_dim)
            xavier_normal_(self.rel_embed_w.weight)

        self.bceloss = torch.nn.BCELoss()
        self.position_embeddings = torch.nn.Embedding(2, self.p.embed_dim)
        xavier_normal_(self.position_embeddings.weight)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)

        self.w_sub = get_param((self.p.embed_dim * 2, 1))
        self.w_rel = get_param((self.p.embed_dim * 2, 1))

        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        encoder_layers = TransformerEncoderLayer(self.p.embed_dim, self.p.heads, self.p.hidden_dim, self.p.hidden_drop)
        self.encoder = TransformerEncoder(encoder_layers, self.p.layers)

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))  # num_entities

        if self.p.method == 'perceptual':
            self.mlp = torch.nn.Linear(self.p.embed_dim * 2, self.p.embed_dim)
        elif self.p.method == 'co-aware':
            self.w_method = get_param((self.p.embed_dim, 1))
            self.w_sub_aware = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
            self.w_rel_aware = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)

        print('Transformer_Triple')

    def loss(self, pred, true_label=None):
        loss = self.bceloss(pred, true_label)
        return loss

    def agg(self, sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N, sub_emb_enlarge=0, rel_emb_enlarge=0):

        if self.p.agg_func == 'HyperCrossE':
            quals_ent_emb, quals_rel_emb = HyperCrossE(sub_emb_enlarge, rel_emb_enlarge, quals_ent_emb, quals_rel_emb)

        sub_score = self.att(sub_emb, quals_rel_emb, N, True)
        rel_score = self.att(rel_emb, quals_rel_emb, N, False)

        emb = self.common(sub_score, rel_score, quals_ent_emb, self.p.method)

        sub = sub_emb * self.p.alpha + (1 - self.p.alpha) * emb
        rel = rel_emb * self.p.alpha + (1 - self.p.alpha) * emb

        return sub, rel

    def att(self, triple_emb, score_quals, N, is_sub):
        triple_emb = triple_emb.unsqueeze(1).repeat(1, score_quals.size(1), 1)
        all_message = torch.cat([triple_emb, score_quals], dim=-1)

        if (is_sub):
            score = torch.matmul(all_message, self.w_sub).squeeze(-1)  # 128,4
        else:
            score = torch.matmul(all_message, self.w_rel).squeeze(-1)

        score = -self.leakyrelu(score)
        num_inf = torch.full_like(score, -np.inf)
        score = torch.where(N.byte(), score, num_inf)

        score = torch.exp(score)
        score_all = score.sum(dim=-1)
        score_all[score_all == 0.0] = 1.0
        score = score.div(score_all.unsqueeze(-1))
        return score

    def common(self, sub_score, rel_score, quals_ent_emb, fusion='co-aware'):

        sub_score = sub_score.unsqueeze(-1).repeat(1, 1, self.p.embed_dim)
        sub_aware_emb = quals_ent_emb * sub_score
        sub_aware_emb = sub_aware_emb.sum(dim=1)

        rel_score = rel_score.unsqueeze(-1).repeat(1, 1, self.p.embed_dim)
        rel_aware_emb = quals_ent_emb * rel_score
        rel_aware_emb = rel_aware_emb.sum(dim=1)

        if fusion == 'mult':
            return sub_aware_emb * rel_aware_emb
        elif fusion == 'perceptual':
            score_emb = torch.cat([sub_aware_emb, rel_aware_emb], dim=-1)
            return self.mlp(score_emb)
        elif fusion == 'co-aware':
            sub_emb_score = torch.matmul(sub_aware_emb, self.w_method).squeeze(-1)
            rel_emb_score = torch.matmul(rel_aware_emb, self.w_method).squeeze(-1)
            sub_emb_score = -self.leakyrelu(sub_emb_score)
            rel_emb_score = -self.leakyrelu(rel_emb_score)

            sub_emb_score = torch.exp(sub_emb_score)
            rel_emb_score = torch.exp(rel_emb_score)
            score_all = sub_emb_score + rel_emb_score
            sub_emb_score = (sub_emb_score / score_all).unsqueeze(-1).repeat(1, self.p.embed_dim)
            rel_emb_score = (rel_emb_score / score_all).unsqueeze(-1).repeat(1, self.p.embed_dim)
            return sub_emb_score * sub_aware_emb + rel_emb_score * rel_aware_emb

    def concat(self, sub_emb, rel_emb):
        ent_embed = sub_emb.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_emb.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([ent_embed, rel_embed], 1).transpose(1, 0)  # [2, batch, embed_dim]
        return stack_inp

    def forward(self, sub, rel, quals, N):
        self.ent_embed.weight.data[0] = 0
        self.rel_embed.weight.data[0] = 0
        self.ent_embed_w.weight.data[0] = 0
        self.rel_embed_w.weight.data[0] = 0

        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)  # batch_size,100

        quals_ent_emb = self.ent_embed(quals[:, 1::2])
        quals_rel_emb = self.rel_embed(quals[:, 0::2])

        if self.p.agg_func == 'HyperCrossE':
            sub_emb_enlarge = self.ent_embed_w(sub).unsqueeze(1).repeat(1, quals_rel_emb.size(1), 1)
            rel_emb_enlarge = self.rel_embed_w(rel).unsqueeze(1).repeat(1, quals_rel_emb.size(1), 1)
            sub_emb, rel_emb = self.agg(sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N, sub_emb_enlarge,
                                        rel_emb_enlarge)
        else:
            sub_emb, rel_emb = self.agg(sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N)

        stk_inp = self.concat(sub_emb, rel_emb)
        mask = torch.zeros((sub.shape[0], 2)).bool().to(self.p.device)  # [128, 2]

        positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.p.device).repeat(stk_inp.shape[1],
                                                                                                  1)  # [128, 2]
        pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
        stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)  # [2, 128, 200]
        x = torch.mean(x, dim=0)

        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))  # batch_size,num_entities
        x += self.bias.expand_as(x)  # batch_size,num_entities

        pred = torch.sigmoid(x)  # batch_size,num_entities

        return pred


class Transformer_Query(torch.nn.Module):
    def __init__(self, params):
        super(Transformer_Query, self).__init__()

        self.p = params

        self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim)  # # batch_size,400
        xavier_normal_(self.ent_embed.weight)
        self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2 + 1, self.p.embed_dim)
        xavier_normal_(self.rel_embed.weight)

        if self.p.agg_func == 'HyperCrossE':
            self.ent_embed_w = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim)  # # batch_size,400
            xavier_normal_(self.ent_embed_w.weight)
            self.rel_embed_w = torch.nn.Embedding(self.p.num_rel * 2 + 1, self.p.embed_dim)
            xavier_normal_(self.rel_embed_w.weight)

        self.bceloss = torch.nn.BCELoss()

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)

        self.w_sub = get_param((self.p.embed_dim * 2, 1))
        self.w_rel = get_param((self.p.embed_dim * 2, 1))

        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        encoder_layers = TransformerEncoderLayer(self.p.embed_dim, self.p.heads, self.p.hidden_dim, self.p.hidden_drop)
        self.encoder = TransformerEncoder(encoder_layers, self.p.layers)

        self.position_embeddings = torch.nn.Embedding(self.p.max_quals + 2, self.p.embed_dim)
        xavier_normal_(self.position_embeddings.weight)

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))  # num_entities

        if self.p.method == 'perceptual':
            self.mlp = torch.nn.Linear(self.p.embed_dim * 2, self.p.embed_dim)
        elif self.p.method == 'co-aware':
            self.w_method = get_param((self.p.embed_dim, 1))
            self.w_sub_aware = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
            self.w_rel_aware = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)

        print('Transformer_Query')

    def loss(self, pred, true_label=None):
        loss = self.bceloss(pred, true_label)
        return loss

    def agg(self, sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N, sub_emb_enlarge=0, rel_emb_enlarge=0):

        if self.p.agg_func == 'HyperCrossE':
            quals_ent_emb, quals_rel_emb = HyperCrossE(sub_emb_enlarge, rel_emb_enlarge, quals_ent_emb, quals_rel_emb)

        sub_score = self.att(sub_emb, quals_rel_emb, N, True)
        rel_score = self.att(rel_emb, quals_rel_emb, N, False)

        emb = self.common(sub_score, rel_score, quals_ent_emb, self.p.method)

        sub = sub_emb * self.p.alpha + (1 - self.p.alpha) * emb
        rel = rel_emb * self.p.alpha + (1 - self.p.alpha) * emb

        return sub, rel

    def att(self, triple_emb, score_quals, N, is_sub):
        triple_emb = triple_emb.unsqueeze(1).repeat(1, score_quals.size(1), 1)
        all_message = torch.cat([triple_emb, score_quals], dim=-1)

        if (is_sub):
            score = torch.matmul(all_message, self.w_sub).squeeze(-1)  # 128,4
        else:
            score = torch.matmul(all_message, self.w_rel).squeeze(-1)

        score = -self.leakyrelu(score)
        num_inf = torch.full_like(score, -np.inf)
        score = torch.where(N.byte(), score, num_inf)

        score = torch.exp(score)
        score_all = score.sum(dim=-1)
        score_all[score_all == 0.0] = 1.0
        score = score.div(score_all.unsqueeze(-1))
        return score

    def common(self, sub_score, rel_score, quals_ent_emb, fusion='co-aware'):

        sub_score = sub_score.unsqueeze(-1).repeat(1, 1, self.p.embed_dim)
        sub_aware_emb = quals_ent_emb * sub_score
        sub_aware_emb = sub_aware_emb.sum(dim=1)

        rel_score = rel_score.unsqueeze(-1).repeat(1, 1, self.p.embed_dim)
        rel_aware_emb = quals_ent_emb * rel_score
        rel_aware_emb = rel_aware_emb.sum(dim=1)

        if fusion == 'mult':
            return sub_aware_emb * rel_aware_emb
        elif fusion == 'perceptual':
            score_emb = torch.cat([sub_aware_emb, rel_aware_emb], dim=-1)
            return self.mlp(score_emb)
        elif fusion == 'co-aware':
            sub_emb_score = torch.matmul(sub_aware_emb, self.w_method).squeeze(-1)
            rel_emb_score = torch.matmul(rel_aware_emb, self.w_method).squeeze(-1)
            sub_emb_score = -self.leakyrelu(sub_emb_score)
            rel_emb_score = -self.leakyrelu(rel_emb_score)

            sub_emb_score = torch.exp(sub_emb_score)
            rel_emb_score = torch.exp(rel_emb_score)
            score_all = sub_emb_score + rel_emb_score
            sub_emb_score = (sub_emb_score / score_all).unsqueeze(-1).repeat(1, self.p.embed_dim)
            rel_emb_score = (rel_emb_score / score_all).unsqueeze(-1).repeat(1, self.p.embed_dim)
            return sub_emb_score * sub_aware_emb + rel_emb_score * rel_aware_emb

    def concat(self, sub_emb, rel_emb, quals_ent_emb, quals_rel_emb):
        ent_embed = sub_emb.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_emb.view(-1, 1, self.p.embed_dim)

        quals = torch.cat((quals_ent_emb, quals_rel_emb), 2).view(-1, 2 * quals_ent_emb.shape[1],
                                                                  quals_rel_emb.shape[2])
        stack_inp = torch.cat([ent_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals, N):
        self.ent_embed.weight.data[0] = 0
        self.rel_embed.weight.data[0] = 0
        self.ent_embed_w.weight.data[0] = 0
        self.rel_embed_w.weight.data[0] = 0

        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)  # batch_size,100

        quals_ent_emb = self.ent_embed(quals[:, 1::2])
        quals_rel_emb = self.rel_embed(quals[:, 0::2])

        if self.p.agg_func == 'HyperCrossE':
            sub_emb_enlarge = self.ent_embed_w(sub).unsqueeze(1).repeat(1, quals_rel_emb.size(1), 1)
            rel_emb_enlarge = self.rel_embed_w(rel).unsqueeze(1).repeat(1, quals_rel_emb.size(1), 1)
            sub_emb, rel_emb = self.agg(sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N, sub_emb_enlarge,
                                        rel_emb_enlarge)
        else:
            sub_emb, rel_emb = self.agg(sub_emb, rel_emb, quals_ent_emb, quals_rel_emb, N)

        mask = torch.zeros((sub.shape[0], N.shape[1] * 2 + 2)).bool().to(self.p.device)  # [128, 2]
        mask[:, 2::2] = N == 0
        mask[:, 3::2] = N == 0
        stk_inp = self.concat(sub_emb, rel_emb, quals_ent_emb, quals_rel_emb)

        positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.p.device).repeat(stk_inp.shape[1],
                                                                                                  1)  # [128, 2]
        pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
        stk_inp = stk_inp + pos_embeddings
        x = self.encoder(stk_inp, src_key_padding_mask=mask)  # [2, 128, 200]
        x = torch.mean(x, dim=0)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))  # batch_size,num_entities
        x += self.bias.expand_as(x)  # batch_size,num_entities
        pred = torch.sigmoid(x)  # batch_size,num_entities
        return pred
