import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from models.utils import log_sum_exp
import numpy as np
from scipy.stats import ortho_group
from torch.distributions.multivariate_normal import MultivariateNormal

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class GaussianEncoderBase(nn.Module):
    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def sample(self, inputs, nsamples):
        mu_c, logvar_c, mu_s, logvar_s = self.forward(inputs)
        c = self.reparameterize(mu_c, logvar_c, nsamples)
        s = self.reparameterize(mu_s, logvar_s, nsamples)
        return c, s, (mu_c, logvar_c), (mu_s, logvar_s)

    def encode(self, inputs, nsamples=1):
        mu_c, logvar_c, mu_s, logvar_s = self.forward(inputs)
        c = self.reparameterize(mu_c, logvar_c, nsamples)
        s = self.reparameterize(mu_s, logvar_s, nsamples)

        KL = 0.5 * (mu_c.pow(2) + logvar_c.exp() - logvar_c - 1).sum(1) + 0.5 * (
                    mu_s.pow(2) + logvar_s.exp() - logvar_s - 1).sum(1)
        return c, s, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(0).expand(nsamples, batch_size, nz)
        std_expd = std.unsqueeze(0).expand(nsamples, batch_size, nz)

        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def sample_from_inference(self, x, nsamples=1):
        assert False
        mu, logvar = self.forward(x)
        batch_size, nz = mu.size()
        return mu.unsqueeze(0).expand(nsamples, batch_size, nz)

    def eval_inference_dist(self, x, z, param=None):
        assert False
        nz = z.size(2)
        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()
        dev = z - mu

        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density.squeeze(0)

    def calc_mi(self, x):
        mu, logvar = self.forward(x)

        x_batch, nz = mu.size()

        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        z_samples = self.reparameterize(mu, logvar, 1)

        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        dev = z_samples - mu

        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        log_qz = log_sum_exp(log_density, dim=0) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()


class LSTMEncoder(GaussianEncoderBase):
    def __init__(self, ni, nh, nc, ns, attention_heads, vocab, device, model_init, emb_init):
        super(LSTMEncoder, self).__init__()
        self.embed = nn.Embedding(len(vocab), ni).from_pretrained(torch.tensor(vocab.glove_embed, device=device, dtype=torch.float), freeze=False)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=2,
                            bidirectional=True)
        self.attention_layer = nn.MultiheadAttention(nh, attention_heads)
        self.linear_c = nn.Linear(nh, 2 * nc, bias=False)
        self.linear_s = nn.Linear(nh, 2 * ns, bias=False)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs):
        # if len(inputs.size()) > 2:
        #     assert False
        #     word_embed = torch.matmul(inputs, self.embed.weight)
        # else:
        #     word_embed = self.embed(inputs)

        word_embed = self.embed(inputs)
        outputs, (last_state, last_cell) = self.lstm(word_embed)
        seq_len, bsz, hidden_size = outputs.size()
        hidden_repr = outputs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]
        # outputs, _ = self.attention_layer(hidden_repr, hidden_repr, hidden_repr)
        # hidden_repr = torch.max(outputs, 0)[0]

        mean_c, logvar_c = self.linear_c(hidden_repr).chunk(2, -1)
        mean_s, logvar_s = self.linear_s(hidden_repr).chunk(2, -1)
        return mean_c, logvar_c, mean_s, logvar_s


class StyleClassifier(nn.Module):
    def __init__(self, ns, num_labels):
        super(StyleClassifier, self).__init__()
        self.linear = nn.Linear(ns, 2**num_labels, bias=False)

    def forward(self, inputs):
        output = self.linear(inputs)
        # output = nn.Softmax(output)
        return output


class ContentDecoder(nn.Module):
    def __init__(self, ni, nc, nh, vocab, model_init, emb_init, device, dropout_in=0, dropout_out=0):
        super(ContentDecoder, self).__init__()
        self.vocab = vocab
        self.device = device

        # self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1)
        self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1).from_pretrained(torch.tensor(vocab.glove_embed, device=device, dtype=torch.float), freeze=False)

        self.lstm = nn.LSTM(input_size=nc + ni,
                            hidden_size=nh,
                            num_layers=1,
                            bidirectional=False)

        self.pred_linear = nn.Linear(nh, len(vocab), bias=False)
        self.trans_linear = nn.Linear(nc, nh, bias=False)

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, c):
        n_sample_c, batch_size_c, nc = c.size()

        seq_len = inputs.size(0)

        word_embed = self.embed(inputs)
        word_embed = self.dropout_in(word_embed)

        if n_sample_c == 1:
            c_ = c.expand(seq_len, batch_size_c, nc)
        else:
            raise NotImplementedError

        word_embed = torch.cat((word_embed, c_), -1)

        c = c.view(batch_size_c * n_sample_c, nc)

        c_init = self.trans_linear(c).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits.view(-1, batch_size_c, len(self.vocab))

    def decode(self, c, greedy=True):
        n_sample_c, batch_size_c, nc = c.size()

        assert (n_sample_c == 1)

        c = c.view(batch_size_c * n_sample_c, nc)

        batch_size = c.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.trans_linear(c).unsqueeze(0)
        h_init = torch.tanh(c_init)
        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long,
                                     device=self.device).unsqueeze(0)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long,
                                  device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:
            word_embed = self.embed(decoder_input)

            word_embed = torch.cat((word_embed, c.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(0)

            if greedy:
                select_index = torch.argmax(output_logits, dim=1)
            else:
                sample_prob = F.softmax(output_logits, dim=1)
                select_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = select_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(select_index[i].item()))

            mask = torch.mul((select_index != end_symbol), mask)

        return decoded_batch


class SgivenC(nn.Module):
    def __init__(self, nc, ns, model_init):
        super(SgivenC, self).__init__()
        self.linear1 = nn.Linear(nc, 128, bias=False)
        self.linear2 = nn.Linear(128, 2*ns, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters(model_init)

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, c):
        output = self.relu(self.linear1(c))
        mean_s, logvar_s = self.linear2(output).chunk(2, -1)

        return mean_s, logvar_s

    def get_log_prob_single(self, s, mean_s, logvar_s):
        cov_s = torch.diag(logvar_s.exp())
        m = MultivariateNormal(mean_s, covariance_matrix=cov_s)

        return m.log_prob(s)

    def get_log_prob_total(self, s, c):
        # print("s_given_c WEIGHTS:", self.linear2.weight.data)
        mean_s, logvar_s = self.forward(c) # NxD
        batch_size = mean_s.size(0)
        cov_s = torch.diag_embed(logvar_s.exp() + 1e-6) # NxDxD
        # cov_s[torch.arange(mean_s.size(1)), torch.arange(mean_s.size(1))] += 1e-6
        m = MultivariateNormal(mean_s, covariance_matrix=cov_s)

        kernel = m.log_prob(s.unsqueeze(1))

        return kernel.trace() / batch_size
        # loss = 0
        # for i in range(batch_size):
        #     loss += self.get_log_prob_single(s[i], mean_s[i], logvar_s[i])
        # return loss/batch_size

    def get_log_prob_total_distribution(self, mean_s_original, logvar_s_original, c):
        mean_s, logvar_s = self.forward(c) # NxD
        batch_size = mean_s.size(0)
        loss = nn.MSELoss(reduction='sum')
        
        return (loss(mean_s, mean_s_original) + loss(logvar_s.exp(), logvar_s_original.exp())) / batch_size

    def get_s_c_mi(self, s, c):
        n_sample_c, batch_size_c, nc = c.size()
        n_sample_s, batch_size_s, ns = s.size()

        c = c.view(batch_size_c * n_sample_c, nc)
        s = s.view(batch_size_s * n_sample_s, ns)

        mean_s, logvar_s = self.forward(c) # NxD
        batch_size = mean_s.size(0)
        cov_s = torch.diag_embed(logvar_s.exp() + 1e-6) # NxDxD
        # cov_s[torch.arange(mean_s.size(1)), torch.arange(mean_s.size(1))] += 1e-6
        m = MultivariateNormal(mean_s, covariance_matrix=cov_s)
        kernel = m.log_prob(s.unsqueeze(1))
        # print("MEAN, COV:", mean_s, cov_s)
        # print("KERNEL:", kernel[0])

        rand_idx = torch.randint(batch_size, (batch_size,))
        mask_tensor = torch.zeros_like(kernel)
        assert (mask_tensor.size()[0] == batch_size and mask_tensor.size()[1] == batch_size)
        mask_tensor[torch.arange(batch_size), rand_idx] = -1
        mask_tensor[torch.arange(batch_size), torch.arange(batch_size)] += 1

        masked_kernel = torch.mul(kernel, mask_tensor)

        return masked_kernel.sum() / batch_size

        # loss = 0
        # for i in range(batch_size):
        #     j = torch.randint(low=0, high=batch_size, size=(1,))[0]
        #     loss += self.get_log_prob_single(s[i], mean_s[i], logvar_s[i]) - self.get_log_prob_single(s[i], mean_s[j], logvar_s[j])

        # return loss/batch_size

class LSTMDecoder(nn.Module):
    def __init__(self, ni, nc, ns, nh, vocab, model_init, emb_init, device, dropout_in=0, dropout_out=0):
        super(LSTMDecoder, self).__init__()
        self.vocab = vocab
        self.device = device

        self.lstm = nn.LSTM(input_size=ni + ns + nc,
                            hidden_size=nh,
                            num_layers=1,
                            bidirectional=False)

        self.pred_linear = nn.Linear(nh, len(vocab), bias=False)
        self.trans_linear = nn.Linear(ns + nc, nh, bias=False)

        self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1).from_pretrained(torch.tensor(vocab.glove_embed, device=device, dtype=torch.float), freeze=False)

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, c, s):
        n_sample_c, batch_size_c, nc = c.size()
        n_sample_s, batch_size_s, ns = s.size()

        assert (n_sample_c == n_sample_s)
        assert (batch_size_c == batch_size_s)

        seq_len = inputs.size(0)

        word_embed = self.embed(inputs)
        word_embed = self.dropout_in(word_embed)

        if n_sample_c == 1:
            c_ = c.expand(seq_len, batch_size_c, nc)
            s_ = s.expand(seq_len, batch_size_s, ns)
        else:
            raise NotImplementedError

        word_embed = torch.cat((word_embed, c_, s_), -1)

        c = c.view(batch_size_c * n_sample_c, nc)
        s = s.view(batch_size_s * n_sample_s, ns)

        concat_c_s = torch.cat((c, s), 1)

        # c_init = self.trans_linear(concat_c_s).unsqueeze(0).repeat(2, 1, 1)
        c_init = self.trans_linear(concat_c_s).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits.view(-1, batch_size_c, len(self.vocab))

    def decode(self, c, s, greedy=True):
        assert False 

        n_sample_c, batch_size_c, nc = c.size()
        n_sample_s, batch_size_s, ns = s.size()

        assert (n_sample_c == 1 and n_sample_s == 1)

        c = c.view(batch_size_c * n_sample_c, nc)
        s = s.view(batch_size_s * n_sample_s, ns)

        z = torch.cat((c, s), 1)
        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.trans_linear(z).unsqueeze(0).repeat(2, 1, 1)
        h_init = torch.tanh(c_init)
        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long,
                                     device=self.device).unsqueeze(0)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long,
                                  device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:
            word_embed = self.embed(decoder_input)

            word_embed = torch.cat((word_embed, z.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(0)

            if greedy:
                select_index = torch.argmax(output_logits, dim=1)
            else:
                sample_prob = F.softmax(output_logits, dim=1)
                select_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = select_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(select_index[i].item()))

            mask = torch.mul((select_index != end_symbol), mask)

        return decoded_batch

    def beam_search_decode(self, z1, z2=None, K=5, max_t=20):
        decoded_batch = []
        if z2 is not None:
            n_sample_c, batch_size_c, nc = z1.size()
            n_sample_s, batch_size_s, ns = z2.size()
            z1 = z1.view(n_sample_c * batch_size_c, nc)
            z2 = z2.view(n_sample_s * batch_size_s, ns)
            z = torch.cat([z1, z2], -1)
        else:
            z = z1
        batch_size, nz = z.size()

        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        for idx in range(batch_size):
            decoder_input = torch.tensor([[self.vocab["<s>"]]], dtype=torch.long,
                                         device=self.device)
            decoder_hidden = (h_init[:, idx, :].unsqueeze(1), c_init[:, idx, :].unsqueeze(1))
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0.1, 1)
            live_hypotheses = [node]

            completed_hypotheses = []

            t = 0
            while len(completed_hypotheses) < K and t < max_t:
                t += 1

                decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=1)

                decoder_hidden_h = torch.cat([node.h[0] for node in live_hypotheses], dim=1)
                decoder_hidden_c = torch.cat([node.h[1] for node in live_hypotheses], dim=1)

                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

                word_embed = self.embed(decoder_input)
                word_embed = torch.cat((word_embed, z[idx].view(1, 1, -1).expand(
                    1, len(live_hypotheses), nz)), dim=-1)

                output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

                output_logits = self.pred_linear(output)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor([node.logp for node in live_hypotheses],
                                         dtype=torch.float, device=self.device)
                decoder_output = decoder_output + prev_logp.view(1, len(live_hypotheses), 1)

                decoder_output = decoder_output.view(-1)

                log_prob, indexes = torch.topk(decoder_output, K - len(completed_hypotheses))

                live_ids = indexes // len(self.vocab)
                word_ids = indexes % len(self.vocab)

                live_hypotheses_new = []
                for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                    node = BeamSearchNode((
                        decoder_hidden[0][:, live_id, :].unsqueeze(1),
                        decoder_hidden[1][:, live_id, :].unsqueeze(1)),
                        live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)

                    if word_id.item() == self.vocab["</s>"]:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)

                live_hypotheses = live_hypotheses_new

                if len(completed_hypotheses) == K:
                    break

            for live in live_hypotheses:
                completed_hypotheses.append(live)

            utterances = []
            for n in sorted(completed_hypotheses, key=lambda node: node.logp, reverse=True):
                utterance = []
                utterance.append(self.vocab.id2word(n.wordid.item()))
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(self.vocab.id2word(n.wordid.item()))

                utterance = utterance[::-1]
                utterances.append(utterance)

                break

            decoded_batch.append(utterances[0])

        return decoded_batch
