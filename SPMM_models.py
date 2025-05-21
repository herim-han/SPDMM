from xbert import BertConfig, BertForMaskedLM
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed
import pytorch_lightning as pl
from scheduler import create_scheduler
import random
import numpy as np

def extract_feature(model, linear_proj, queue, inputs, is_momentum=False): #default=student
    def forward():
        embeds = model(**inputs, return_dict=True).last_hidden_state
        feat = F.normalize(linear_proj(embeds[:, 0, :]), dim=-1) #prop_feat
        feat_all = torch.cat([feat.t(), queue.clone().detach()], dim=1) #feat_all
#        print('extract_feature.forward', self.property_encoder is model, self.property_proj is linear_proj)
        return embeds, feat, feat_all

    if is_momentum:
        with torch.no_grad():
            return forward()
    else:
        return forward()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SPMM(pl.LightningModule):
    #def __init__(self, tokenizer=None, config=None, loader_len=0, no_train=False):
    def __init__(self, tokenizer=None, config=None, loader_len=0, no_train=False,debugging=False):
        super().__init__()
        self.debugging=debugging
        self.automatic_optimization = False
        self.config = config
        self.tokenizer = tokenizer
        self.training_step_outputs = []

        embed_dim = config['embed_dim']

        #unimodal encoder for SMILES
        bert_config = BertConfig.from_json_file(config['bert_config_text'])#load from config_bert.json
        self.text_encoder = BertForMaskedLM(config=bert_config)
        text_width = self.text_encoder.config.hidden_size
        property_width = text_width

        self.property_proj = nn.Linear(property_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width * 2, 2)
        #unimodal encoder for PV
        self.property_embed = nn.Linear(1, property_width)
        bert_config2 = BertConfig.from_json_file(config['bert_config_property'])#load from config_bert_property.json
        self.property_encoder = BertForMaskedLM(config=bert_config2).bert
        self.property_mtr_head = nn.Sequential(nn.Linear(property_width, property_width),
                                               nn.GELU(),
                                               nn.LayerNorm(property_width, bert_config.layer_norm_eps),
                                               nn.Linear(property_width, 1))
        self.property_cls = nn.Parameter(torch.zeros(1, 1, property_width))
        self.property_mask = nn.Parameter(torch.zeros(1, 1, property_width))    # unk token for PV

        # create momentum models (knowledge distillation providing pseudo-label)
        self.property_encoder_m = BertForMaskedLM(config=bert_config2).bert
        self.property_proj_m = nn.Linear(property_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        # not update teacher model
        for p in self.property_encoder_m.parameters():  p.requires_grad = False
        for p in self.property_proj_m.parameters():     p.requires_grad = False
        for p in self.text_encoder_m.parameters():      p.requires_grad = False
        for p in self.text_proj_m.parameters():         p.requires_grad = False
        #pair for student model, teacher model
        self.model_pairs = [[self.property_encoder, self.property_encoder_m],
                            [self.property_proj, self.property_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.text_config = bert_config
        self.prop_config = bert_config2

        self.copy_params()

        # create the queue
        if not no_train:
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            self.mlm_probability = config['mlm_probability']
            self.warmup_steps = config['schedular']['warmup_epochs']
            self.loader_len = loader_len
            self.momentum = config['momentum']
            self.queue_size = config['queue_size']
            self.register_buffer("prop_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.prop_queue = nn.functional.normalize(self.prop_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, property_original, text_input_ids, text_attention_mask, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5)
#        print(f'prop config \n {self.prop_config}')
#        print(f'text config \n {self.text_config}')
#        exit(-1)
        property_feature = self.property_embed(property_original.unsqueeze(2))
        #print(property_feature.shape)

        unk_tokens = self.property_mask.expand(property_original.size(0), property_original.size(1), -1) # "learnable" empty torch (B, L_prop, width)
        #print('unk tokens', unk_tokens.shape)
        mpm_mask = torch.bernoulli(torch.ones_like(property_original) * 0.5)    # 1 for mask, 0 for keep, 0.5=masking ratio (B, L)
        mpm_mask_expand = mpm_mask.unsqueeze(2).repeat(1, 1, unk_tokens.size(2)) # matching dimension: (B, L) > (B, L, 1) > (B, L, width)
        property_masked = property_feature * (1 - mpm_mask_expand) + unk_tokens * mpm_mask_expand # replaced masked tokens

        properties = torch.cat([self.property_cls.expand(property_original.size(0), -1, -1), property_masked], dim=1) # add "learnable" cls (B, L+1, width)
        
        prop_embeds, prop_feat, _ = extract_feature(
                                                     self.property_encoder,
                                                     self.property_proj,
                                                     self.prop_queue,
                                                     {"inputs_embeds": properties},
                                                     is_momentum=False, 
                                                   )

        prop_atts = torch.ones(prop_embeds.size()[:-1], dtype=torch.long).to(properties.device) # (B, L+1)

        text_embeds, text_feat, _ = extract_feature(
                                                     self.text_encoder.bert,
                                                     self.text_proj,
                                                     self.text_queue,
                                                     {"input_ids": text_input_ids,
                                                      "attention_mask": text_attention_mask,
                                                      "mode":'text'},
                                                     is_momentum=False, 
                                                   )
        
        # get momentum features generating soft target (probability)
        with torch.no_grad():
            self._momentum_update()

            prop_embeds_m, prop_feat_m, prop_feat_all = extract_feature(
                                                                         self.property_encoder_m,
                                                                         self.property_proj_m,
                                                                         self.prop_queue,
                                                                         {"inputs_embeds": properties},
                                                                         is_momentum=True,
                                                                        )
#            print('2222222', prop_embeds_m.shape, prop_feat_m.shape, prop_feat_all.shape)

            text_embeds_m, text_feat_m, text_feat_all = extract_feature(
                                                                         self.text_encoder_m.bert,
                                                                         self.text_proj_m,
                                                                         self.text_queue,
                                                                         {"input_ids": text_input_ids,
                                                                          "attention_mask": text_attention_mask,
                                                                          'mode':'text'},
                                                                         is_momentum=True, 
                                                                       )
#            print('2222222', text_embeds_m.shape, text_feat_m.shape, text_feat_all.shape)

        #inter-modality(p -> t, t -> p)
        sim_i2t, sim_t2i,loss_i2t,loss_t2i = self.contrastive_loss(prop_feat, text_feat, 
                                                                    prop_feat_m, text_feat_m, 
                                                                    prop_feat_all, text_feat_all, 
                                                                    alpha, self.temp)
        #intra-modality(p -> p, t -> t)
        sim_i2i, sim_t2t,loss_i2i,loss_t2t = self.contrastive_loss(prop_feat, text_feat, 
                                                                    prop_feat_m, text_feat_m, 
                                                                    text_feat_all, prop_feat_all, 
                                                                    alpha, self.temp)
        loss_ita = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 2

        if torch.isnan(sim_i2t).any() or torch.isnan(sim_t2i).any() or torch.isnan(loss_ita):
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        # ================ ITM: Image-Text Matching ================= #
        # forward the positve image(prop)-text pair
        # cross attention: Q=prop, K,V=text
        pos_pos = self.cross_attention_pair(prop_embeds, prop_atts, text_embeds, text_attention_mask)

        #hard negative mining: trained more using high similarity with negative sample
        with torch.no_grad():
            bs = properties.size(0) #current Batch (or current prop sequence order)
            # hard
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            # weights_i2t[b][b]=0 : b-th prop to b-th text = 0
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        prop_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item() #stochastic sampling (torch.argmax for deterministic sampling)
            prop_embeds_neg.append(prop_embeds[neg_idx]) #negative sample "slice" (1, L, width) from prop_embeds = (B, L, width=768)
        prop_embeds_neg = torch.stack(prop_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # positive + negative sample feature
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_attention_mask, text_atts_neg], dim=0)
        # negative + positive sample feature
        prop_embeds_all = torch.cat([prop_embeds_neg, prop_embeds], dim=0)
        prop_atts_all = torch.cat([prop_atts, prop_atts], dim=0)

        # cross attention: additional update using combination of positive and negative sample 
        # encoder-only (bert, using full tokens, bidirectional context, extract features)
        pos_neg = self.cross_attention_pair(prop_embeds_all, prop_atts_all, text_embeds_all, text_atts_all)

        # positive-positive pair (B, 2D), positive-negative pair (2*B, 2D)
        vl_embeddings = torch.cat([pos_pos, pos_neg], dim=0)
        # binary classification for pair (2*B, 2) itm_head=MLP head (nn.Linear)
        vl_output = self.itm_head(vl_embeddings)
        # True label for predicting pair
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(properties.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        # adding queue from momentum teacher feature in the Batch
        #print('11111111', prop_feat_m.shape, text_feat_m.shape)
        self._dequeue_and_enqueue(prop_feat_m, text_feat_m)

        # ================= MLM: Masked Language Modeling + teacher distillation ================= #
        # Auto-regressive text prediction
        input_ids = text_input_ids.clone() #(B, L)
        labels = input_ids.clone()[:, 1:] # (B, L-1), shifted target (target for t+1)

        with torch.no_grad():
        # Decoder-style, Language Modeling (causal, no text-masking)
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text_attention_mask,
                                           encoder_hidden_states=prop_embeds_m, #Q, V (prop for conditioning context)
                                           encoder_attention_mask=prop_atts,
                                           return_dict=True,
                                           is_decoder=True, #cross attention + causal mask
                                           return_logits=True,
                                           )[:, :-1, :] #(B, L-1, dim)

        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=prop_embeds,
                                       encoder_attention_mask=prop_atts,
                                       return_dict=True,
                                       is_decoder=True,#1.causal mask=use only previous tokens 2. cross-attention
                                       return_logits=True,
                                       )[:, :-1, :]

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_mlm = loss_fct(mlm_output.permute((0, 2, 1)), labels)

        # kl loss between two distributions (student output-F.log_softmax, teacher output-logits_m)
        loss_distill_text = -torch.sum(F.log_softmax(mlm_output, dim=-1) * F.softmax(logits_m, dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()
        # loss_mlm from student model (hard target) and kl loss (soft target)
        loss_mlm = (1 - alpha) * loss_mlm + alpha * loss_distill_text

        # ================= MPM ================= #
        target = property_original.clone()
        # properties = cls + masked prop sequence, is_decoder: autoregressive prediction (use only previous tokens)
        prop_embeds_causal = self.property_encoder(inputs_embeds=properties, is_decoder=True, return_dict=True).last_hidden_state

        # Encoder-style, Recovery masked prop (non-causal, random)
        prop_output = self.text_encoder.bert(encoder_embeds=prop_embeds_causal, #input: prop
                                             attention_mask=prop_atts,
                                             encoder_hidden_states=text_embeds, #condition: text (for cross attention)
                                             encoder_attention_mask=text_attention_mask,
                                             return_dict=True,
                                             is_decoder=True,#1.causal mask 2. **cross-attention -> but real, non-causal, bidirectional masked pred
                                             mode='fusion',
                                             ).last_hidden_state[:, :-1, :]

        pred = self.property_mtr_head(prop_output).squeeze()

        lossfn = nn.MSELoss()
        # idx slicing from masking (above : mpm_mask = 1 for mask, 0 for keep, 0.5=masking ratio (B, L))
        loss_mpm = lossfn(pred[(1 - mpm_mask).to(bool)], target[(1 - mpm_mask).to(bool)])

        return loss_mlm, loss_mpm * 5, loss_ita, loss_itm

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_feat, text_feat):
        img_feats = concat_all_gather(img_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = img_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.prop_queue[:, ptr:ptr + batch_size] = img_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def configure_optimizers(self):
        arg_opt = self.config['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
        arg_sche = AttrDict(self.config['schedular'])
        # scheduler/scheduler_factory.py (cosine scheduler)
        scheduler, _ = create_scheduler(arg_sche, optimizer)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        print('qqq', metric)

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        prop, text = train_batch #tensor (B, 53), list (B)
        # self.tokenizer = BertTokenizer from transformers
        text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(prop.device)

        #warm up lr
        alpha = self.config['alpha'] if self.current_epoch > 0 else self.config['alpha'] * min(1., batch_idx / self.loader_len)

        # w/ line 289 tokenization_bert.py // return [CLS] + tokens + [SEP]
        #loss_mlm, loss_mpm, loss_ita, loss_itm = self(prop, text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], alpha=alpha)
        # w/ line 288 tokenization_bert.py // return tokens + [SEP]
        loss_mlm, loss_mpm, loss_ita, loss_itm = self(prop, text_input.input_ids, text_input.attention_mask, alpha=alpha)
        loss = loss_mlm + loss_mpm + loss_ita + loss_itm
        if loss != torch.tensor(0.):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            optimizer.step()
        else:
            print('aaaaaaaaaaaa')
        if self.global_rank == 0:
            self.log('lr', optimizer.param_groups[0]["lr"], prog_bar=True)
            self.log('loss_mlm', loss_mlm, prog_bar=True)
            self.log('loss_mpm', loss_mpm, prog_bar=True)
            self.log('loss_ita', loss_ita, prog_bar=True)
            self.log('loss_itm', loss_itm, prog_bar=True)

        step_size = 100
        warmup_iterations = self.warmup_steps * step_size
        if self.current_epoch > 0 and batch_idx == 0: #training phase (lr decay along with cosine curve)
            scheduler.step(self.current_epoch + self.warmup_steps)
        else: #warmup phase (linear increase lr)
            if self.current_epoch == 0 and batch_idx % step_size == 0 and batch_idx <= warmup_iterations:
                scheduler.step(batch_idx // step_size)
        # loss during training step for "batch" not "epoch"
        self.training_step_outputs.append(torch.tensor([loss_mlm, loss_mpm, loss_ita, loss_itm]))
        return torch.tensor([loss_mlm, loss_mpm, loss_ita, loss_itm])

    def on_train_epoch_end(self): # outputs: collection of returns from 'training_step'
        # only consider average loss for last 1000 batches
        tmp = torch.stack(self.training_step_outputs[-1000:]).mean(dim=0).tolist()
        if self.global_rank == 0:
            print(f'\n mean loss: {tmp[0]:.4f}, {tmp[1]:.4f}, {tmp[2]:.4f}, {tmp[3]:.4f}')
        self.training_step_outputs.clear()


    def cross_attention_pair(self, q_embeds, q_atts, k_embeds, k_atts):
        # prop / text cross attention for pair matching
        q_pair = self.text_encoder.bert(
                                           encoder_embeds=q_embeds,
                                           attention_mask=q_atts,
                                           encoder_hidden_states=k_embeds,
                                           encoder_attention_mask=k_atts,
                                           return_dict=True,
                                           mode="fusion"
                                          ).last_hidden_state[:, 0, :]  # (B, D)
    
        k_pair = self.text_encoder.bert(
                                           encoder_embeds=k_embeds,
                                           attention_mask=k_atts,
                                           encoder_hidden_states=q_embeds,
                                           encoder_attention_mask=q_atts,
                                           return_dict=True,
                                           mode="fusion"
                                          ).last_hidden_state[:, 0, :]  # (B, D)

        qk_pair = torch.cat([q_pair, k_pair], dim=-1) 
        return qk_pair #cls token

    def cross_attention_mlm(self, model, inputs, is_momentum=False): #default=student
        def forward():
            mlm_output = model(**inputs
                               )[:, :-1, :] #(B, L-1, dim)
            return mlm_output
    
        if is_momentum:
            with torch.no_grad():
                return forward()
        else:
            return forward()

    def contrastive_loss(self, 
                         q_feat, k_feat,#student 
                         q_feat_m, k_feat_m, #teacher
                         q_feat_all, k_feat_all, #teacher
                         alpha, temp):
        """
        args:
            Q_feat: Tensor of shape (B, D), property features from student model
            Q_feat_m: Tensor of shape (B, D), property features from teacher (momentum) model
            K_feat_all: Tensor of shape (D, B+Bq), all text features from teacher model
            alpha: float, soft vs. hard mixing ratio
            temp: float, temperature scaling
    
        Returns:
            loss_q2k: scalar tensor
        """
    
        with torch.no_grad():
            sim_q2k_m = q_feat_m @ k_feat_all / temp  # (B, B+Bq)
            sim_k2q_m = k_feat_m @ q_feat_all / temp  # (B, B+Bq)
            sim_targets = torch.zeros(sim_q2k_m.size(), device=q_feat.device)
            sim_targets.fill_diagonal_(1)
            sim_q2k_targets = alpha * F.softmax(sim_q2k_m, dim=1) + (1 - alpha) * sim_targets
            sim_k2q_targets = alpha * F.softmax(sim_k2q_m, dim=1) + (1 - alpha) * sim_targets
    
        sim_q2k = q_feat @ k_feat_all / temp
        sim_k2q = k_feat @ q_feat_all / temp
        loss_q2k = -torch.sum(F.log_softmax(sim_q2k, dim=1) * sim_q2k_targets, dim=1).mean()
        loss_k2q = -torch.sum(F.log_softmax(sim_k2q, dim=1) * sim_k2q_targets, dim=1).mean()
    
        return sim_q2k, sim_k2q, loss_q2k, loss_k2q

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
