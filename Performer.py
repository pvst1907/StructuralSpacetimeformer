import torch.nn as nn
import torch
import numpy as np

class Performer(nn.Module):
    def _init_(self):
        super()._init_()

    # L >= M case - after i reaches M, we only use M keys, i.e. the maximum number of keys available
    def case1(self, Q, K, V, sent_embed_slice, qkv_size, masked = False):

        self.qkv_size = qkv_size
        self.masked = masked
        L_ = Q.shape[1]
        M_ = V.shape[1]
        Performer = torch.zeros(sent_embed_slice.shape[0], *(L_, self.qkv_size))

        M = torch.matmul(self.phi(K[:, 0, :]).view(sent_embed_slice.shape[0], self.qkv_size, 1),
                         V[:, 0, :].view(sent_embed_slice.shape[0], 1, self.qkv_size))
        N = self.phi(K[:, 0, :])

        if self.masked:
            for i in range(L_):
                if i < M_:
                    M = M + torch.matmul(self.phi(K[:, i, :]).view(sent_embed_slice.shape[0], self.qkv_size, 1),
                                         V[:, i, :].view(sent_embed_slice.shape[0], 1, self.qkv_size))
                    N = N + self.phi(K[:, i, :])

                    new_x = torch.matmul(self.phi(Q[:, i, :]), M)
                    r = torch.matmul(self.phi(Q[:, i, :]), N.view(sent_embed_slice.shape[0], self.qkv_size, 1))
                    sign = torch.tensor(np.sign(r.detach().numpy()))
                    Performer[:, i, :] = torch.div(new_x, (r + sign * 1e-6)).view(sent_embed_slice.shape[0],
                                                                                  self.qkv_size)
                else:
                    new_x = torch.matmul(self.phi(Q[:, i, :]), M)
                    r = torch.matmul(self.phi(Q[:, i, :]), N.view(sent_embed_slice.shape[0], self.qkv_size, 1))
                    sign = torch.tensor(np.sign(r.detach().numpy()))
                    Performer[:, i, :] = torch.div(new_x, (r + sign * 1e-6)).view(sent_embed_slice.shape[0],
                                                                                  self.qkv_size)

        else:
            sum = torch.zeros(sent_embed_slice.shape[0], self.qkv_size, self.qkv_size)
            norm_sum = torch.zeros(sent_embed_slice.shape[0], self.qkv_size, self.qkv_size)
            for j in range(M_):
                sum += torch.matmul(self.phi(K[:, j, :]).view(sent_embed_slice.shape[0], self.qkv_size, 1),
                                    V[:, j, :].view(sent_embed_slice.shape[0], 1, self.qkv_size))
                norm_sum += self.phi(K[:, j, :]).view(sent_embed_slice.shape[0], self.qkv_size, 1)
            for i in range(L_):
                new_x = torch.matmul(self.phi(Q[:, i, :]).view(sent_embed_slice.shape[0], 1, self.qkv_size), sum)
                r = torch.matmul(self.phi(Q[:, i, :]).view(sent_embed_slice.shape[0], 1, self.qkv_size), norm_sum)
                Performer[:, i, :] = torch.div(new_x, r).view(sent_embed_slice.shape[0], self.qkv_size)

        return Performer

    # L < M case
    def case2(self, Q, K, V, sent_embed_slice, qkv_size, masked = False):
        self.qkv_size = qkv_size
        self.masked = masked
        L_ = Q.shape[1]
        M_ = V.shape[1]
        Performer = torch.zeros(sent_embed_slice.shape[0], *(L_, self.qkv_size))
        n = M_ // L_
        rem = M_ % L_
        for i in range(n):
            K_ = K[:, i*L_:(i+1)*L_, :]
            V_ = V[:, i*L_:(i+1)*L_, :]
            Performer += self.case1(Q, K_, V_, sent_embed_slice, qkv_size, masked = self.masked)

        if rem > 0:
            K_ = K[:, (M_ - rem): rem]
            V_ = V[:, (M_ - rem): rem]
            Performer += self.case1(Q, K_, V_, sent_embed_slice, qkv_size, masked = self.masked)

        return Performer

    def forward(self, Q, K, V, sent_embed_slice, qkv_size, masked=False):
        self.qkv_size = qkv_size
        self.masked = masked
        L_ = Q.shape[1]
        M_ = V.shape[1]

        if L_ >= M_:
            return self.case1(Q, K, V, sent_embed_slice, self.qkv_size, self.masked)
        else:
            return self.case2(Q, K, V, sent_embed_slice, self.qkv_size, self.masked)

    def phi(self, x):
        x = x.view(x.shape[0], x.shape[1], 1)
        d = x.shape[1]  # N_w
        M = self.qkv_size  # s_qkv
        N = torch.randn(M, d)  # (s_qkv x Nw)
        mult = torch.matmul(N, x)  # (s_qkv x 1)
        exp = torch.exp(mult)  # (s_qkv x 1)
        norm = (1 / np.sqrt(M)) * torch.exp(-0.5 * torch.norm(x)) * exp  # (s_qkv x 1)
        norm = norm.view(x.shape[0], 1, -1)  # (1 x s_qkv)
        return norm