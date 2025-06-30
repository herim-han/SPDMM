from xbert import BertConfig, BertForMaskedLM
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed
import pytorch_lightning as pl
from scheduler import create_scheduler
import random
import numpy as np
import copy
import json
import pickle

def create_unimodal_models(config, hidden_width, embed_dim, norm_eps, is_momentum=False):
    encoder = BertForMaskedLM(config=config)
    proj_layer = nn.Linear(hidden_width, embed_dim)
    if is_momentum:
        for p in encoder.parameters():      p.requires_grad = False
        for p in proj_layer.parameters():   p.requires_grad = False
        return encoder, proj_layer
    else:
        mtr_head = nn.Sequential(nn.Linear(hidden_width, hidden_width),
                                                   nn.GELU(),
                                                   nn.LayerNorm(hidden_width, norm_eps),
                                                   nn.Linear(hidden_width, 1),
                                                   nn.Softplus())
        cls_token   = nn.Parameter(torch.zeros(1,1, hidden_width))
        mask_token  = nn.Parameter(torch.zeros(1,1, hidden_width))
        return encoder, mtr_head, cls_token, mask_token, proj_layer

def extract_feature(model, linear_proj, queue, inputs, is_momentum=False): #default=student
    def featurize():
        embeds = model(**inputs, return_dict=True).last_hidden_state
        feat = F.normalize(linear_proj(embeds[:, 0, :]), dim=-1) #prop_feat
        queue_on_device = queue.clone().detach().to(feat.device)
        feat_all = torch.cat([feat.t(), queue_on_device], dim=1) #feat_all
        return embeds, feat, feat_all

    if is_momentum:
        with torch.no_grad():
            return featurize()
    else:
        return featurize()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SPMM(pl.LightningModule):
    def __init__(self, tokenizer=None, config=None, loader_len=0, no_train=False,debugging=False):
        super().__init__()
        self.debugging=debugging
        self.automatic_optimization = False
        self.config = config
        self.tokenizer = tokenizer
        self.training_step_outputs = []

        embed_dim = config['embed_dim']
        hidden_width = config['property_width']

        self.bert_config   =  BertConfig.from_json_file(config['bert_config_text'])
        self.bert_config2  =  BertConfig.from_json_file(config['bert_config_property'])
         
        # create student models (training target)
        self.init_modal("prop", self.bert_config2, hidden_width, embed_dim)
        self.init_modal("text", self.bert_config, hidden_width, embed_dim)
        self.init_modal("dist", self.bert_config2, hidden_width, embed_dim)

        self.prop_encoder = self.prop_encoder.bert
        self.prop_embed = nn.Linear(1, hidden_width)
        self.dist_encoder = self.dist_encoder.bert
        self.dist_embed_layer = Embed3DStruct(hidden_width//2, hidden_width//2)

        self.itm_head = nn.Linear(hidden_width * 2, 2)

        # create momentum models (knowledge distillation providing pseudo-label)
        self.init_modal("prop", self.bert_config2, hidden_width, embed_dim, is_momentum=True)
        self.init_modal("text", self.bert_config, hidden_width, embed_dim, is_momentum=True)
        self.init_modal("dist", self.bert_config2, hidden_width, embed_dim, is_momentum=True)
        self.prop_encoder_m = self.prop_encoder_m.bert
        self.dist_encoder_m = self.dist_encoder_m.bert

        #pair for student model, teacher model
        self.model_pairs = [[self.prop_encoder, self.prop_encoder_m],
                            [self.prop_proj, self.prop_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.dist_encoder, self.dist_encoder_m],
                            [self.dist_proj, self.dist_proj_m],
                            ]

        self.copy_params()

        # create the queue
        if not no_train:
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
#            self.mlm_probability = config['mlm_probability']
            self.warmup_steps = config['schedular']['warmup_epochs']
            self.loader_len = loader_len
            self.momentum = config['momentum']
            self.queue_size = config['queue_size']
            self.register_buffer("prop_queue", torch.randn(embed_dim, self.queue_size) )
            self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size) )
            self.register_buffer("dist_queue", torch.randn(embed_dim, self.queue_size) )
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long) )
            self.prop_queue = nn.functional.normalize(self.prop_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
            self.dist_queue = nn.functional.normalize(self.dist_queue, dim=0)

        self.static_config = {
        "prop_encoder": self.prop_encoder,
        "text_encoder": self.text_encoder,
        "dist_encoder": self.dist_encoder,
        "prop_proj": self.prop_proj,
        "text_proj": self.text_proj,
        "dist_proj": self.dist_proj,
        "prop_queue": self.prop_queue,
        "text_queue": self.text_queue,
        "dist_queue": self.dist_queue,
        "prop_mtr": self.prop_mtr_head,
        "dist_mtr": self.dist_mtr_head,
        }
        self.static_m_config = {
        "prop_encoder_m": self.prop_encoder_m,
        "text_encoder_m": self.text_encoder_m,
        "dist_encoder_m": self.dist_encoder_m,
        "prop_proj_m": self.prop_proj_m,
        "text_proj_m": self.text_proj_m,
        "dist_proj_m": self.dist_proj_m,
        }

    def forward(self, property_original, text_input_ids, text_attention_mask, atom_pair, dist, alpha=0):
        if self.debugging:
            self.set_eval_mode()
        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5) #all elements in range (min, max)

        property_feature = self.prop_embed(property_original.clone().detach().unsqueeze(2))
        unk_tokens = self.prop_mask.expand(property_original.size(0), property_original.size(1), -1)
        prop_mpm_mask = torch.bernoulli(torch.ones_like(property_original) * 0.5) #(B, len)
        mpm_mask_expand = prop_mpm_mask.unsqueeze(2).repeat(1, 1, unk_tokens.size(2)) #(B, len) > (B, len, embed_dim)
        property_masked = property_feature * (1 - mpm_mask_expand) + unk_tokens * mpm_mask_expand #(B, len, embed_dim)
        properties = torch.cat([self.prop_cls.expand(property_original.size(0), -1, -1), property_masked], dim=1) #(B, len+1, embed_dim)
         
        dist_feature = self.dist_embed_layer(atom_pair, dist)
        unk_tokens   = self.dist_mask.expand(dist_feature.size(0), dist_feature.size(1), -1)#(B,len+1,embed_dim)
        dist_mpm_mask  = torch.bernoulli(torch.ones_like(dist) * 0.5)
        mpm_mask_expand = dist_mpm_mask.unsqueeze(2).repeat(1, 1, unk_tokens.size(2))
        dist_masked = dist_feature * (1 - mpm_mask_expand) + unk_tokens * mpm_mask_expand #(B, len, embed_dim)
        distances    = torch.cat([self.dist_cls.expand(dist_feature.size(0), -1,-1), dist_masked], dim=1)#(B,len+1,embed_dim)

        dynamic_inputs = {
        'prop': {"inputs_embeds": properties},
        'text': {"input_ids": text_input_ids,
                 "attention_mask": text_attention_mask,
                 "mode": 'text'},# !!! this mode only for 'text' encoder
        'dist': {"inputs_embeds": distances},
        'is_momentum':False
        }

        model_config = self.build_config(**self.static_config, **dynamic_inputs)
        results= {}
        for modal in model_config.keys():      # modal: prop, text, dist
            e_model = model_config[modal]["model"].bert if modal == 'text' else model_config[modal]["model"]
            embeds, feat, _ = extract_feature(
                e_model, model_config[modal]["proj"], model_config[modal]["queue"], model_config[modal]["inputs"], model_config[modal]["is_momentum"]
            )
            atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
#            print('!!!!!!!!!!!!! features from student models ', embeds.shape, feat.shape)
            if modal == 'text':
                atts = text_attention_mask
            results[modal] = {
                "embeds": embeds,
                "feat": feat,
                "atts": atts
            }
        self._momentum_update()
        dynamic_inputs['is_momentum']=True
        model_m_config = self.build_config(**self.static_m_config, **dynamic_inputs)
        results_m = {}
        for modal in  model_m_config.keys():      # modality: prop, text, dist
            m_encoder = model_m_config[modal]["model"].bert if modal == 'text' else model_m_config[modal]["model"]
            
            m_embeds, m_feat, feat_all = extract_feature(
                m_encoder, model_m_config[modal]["proj"], model_config[modal]["queue"], model_config[modal]["inputs"], is_momentum=True
            )
#            print('!!!!!!!!!!!!! features from momentum models ', m_embeds.shape, m_feat.shape, feat_all.shape)
            results_m[modal] = {
                "m_embeds": m_embeds,
                "m_feat": m_feat,
                "m_feat_all": feat_all
            }

        #inter-modality
        inter_pairs = [ ('prop', 'text'), ('dist', 'text'), ('prop', 'dist') ]
        sim_dict, loss_dict ={}, {} 
        for (res1_key, res2_key) in inter_pairs:
            sim1, sim2, loss1, loss2 = self.contrastive_loss( (results[res1_key]['feat'], 
                                                               results_m[res1_key]['m_feat'], 
                                                               results_m[res1_key]['m_feat_all']),
                                                              (results[res2_key]['feat'], 
                                                               results_m[res2_key]['m_feat'], 
                                                               results_m[res2_key]['m_feat_all']),
                                                              alpha, self.temp, mode='inter' )
            if any(torch.isnan(x).any() for x in [sim1, sim2, loss1, loss2]):
                sim1, sim2, loss1, loss2 = 0, 0, 0 , 0
            sim_dict[f'{res1_key}2{res2_key}'], sim_dict[f'{res2_key}2{res1_key}'] = sim1, sim2
            loss_dict[f'{res1_key}2{res2_key}'],loss_dict[f'{res2_key}2{res1_key}'] =loss1, loss2

        intra_pairs = [ 'prop', 'text', 'dist' ]
        for key in intra_pairs:
            sim_score, loss, = self.contrastive_loss( (results[key]['feat'], 
                                                       results_m[key]['m_feat'], 
                                                       results_m[key]['m_feat_all']), 
                                                      alpha, self.temp, mode='intra')
            if any(torch.isnan(x).any() for x in [sim_score, loss]):
                sim_score, loss = 0 , 0
            sim_dict[f'{key}2{key}'], loss_dict[f'{key}2{key}'] = sim_score, loss

        all_loss  = sum(loss_dict.values() )
        loss_ita = all_loss * 0.5
        
        #Image-Text Matching (ITM)
        loss_itm = {}
        for (res1_key, res2_key) in inter_pairs:
            loss_itm[f'{res1_key}2{res2_key}'] = self.pair_matching(res1_key, res2_key, results, sim_dict)
        self._dequeue_and_enqueue(results_m['prop']['m_feat'], results_m['text']['m_feat'], results_m['dist']['m_feat'])

        #Auto-regressive Text (Conditional MLM + KD)
        input_ids = text_input_ids.clone() #(B, L)
        labels = input_ids.clone()[:, 1:] # (B, L-1), shifted target (target for t+1)
        text_features = (input_ids, labels, results['text']['atts'])
        loss_mlm = {}
        for key in ['prop', 'dist']:
            conditional = (results[key]['embeds'], results[key]['atts'], results_m[key]['m_embeds'])
            loss_mlm[key] = self.mlm_prediction(text_features, conditional, alpha)

        # ================= MPM ================= #
        text_features = (results['text']['embeds'], results['text']['atts'])
        loss_mvm = {}
        for (key, mask, targets) in [('prop', prop_mpm_mask, property_original.clone()), ('dist', dist_mpm_mask, dist.clone())]:
            prop_features = (dynamic_inputs[key]["inputs_embeds"], results[key]['atts'], mask, targets)
            loss_mvm[key] = self.mpm_prediction(text_features, 
                                                prop_features,
                                                model_config[key]["model"], 
                                                model_config[key]["mtr_head"])
        loss_mpm = loss_mvm['prop']
        loss_mdm = loss_mvm['dist']
        return sum(loss_mlm.values()), loss_mpm* 5, loss_mdm*10, loss_ita, sum(loss_itm.values())

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):#student, momentum
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        '''
        param_m = lambda * param_m + (1-lambda)*param
        '''
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):#student, momentum
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_feat, text_feat, dist_feat):
        img_feats = concat_all_gather(img_feat)
        text_feats = concat_all_gather(text_feat)
        dist_feats = concat_all_gather(dist_feat)

        batch_size = img_feats.shape[0]

        ptr = int(self.queue_ptr)
        #print('queue/batch size',self.queue_size, batch_size)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.prop_queue[:, ptr:ptr + batch_size] = img_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.dist_queue[:, ptr:ptr + batch_size] = dist_feats.T
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
        prop, text, atom_pair, dist = train_batch #tensor (B, 53), list (B), list (B)
        # self.tokenizer = BertTokenizer from transformers
        # text_input contained two [CLS] token
        text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(prop.device)

        #warm up lr
        alpha = self.config['alpha'] if self.current_epoch > 0 else self.config['alpha'] * min(1., batch_idx / self.loader_len)

        # w/ line 289 tokenization_bert.py // return [CLS] + tokens + [SEP]
        #loss_mlm, loss_mpm, loss_ita, loss_itm = self(prop, text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], alpha=alpha)
        # w/ line 288 tokenization_bert.py // return tokens + [SEP]
        loss_mlm, loss_mpm,loss_mdm, loss_ita, loss_itm = self(prop, text_input.input_ids, text_input.attention_mask, atom_pair, dist, alpha=alpha)

        loss = loss_mlm + loss_mpm +loss_mdm+ loss_ita + loss_itm
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
            self.log('loss_mdm', loss_mpm, prog_bar=True)
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

    def pair_matching(self, pair1, pair2, results, sim_dict):
        #pair1=prop , pair2=text 
        pos_pos = self.cross_attention_pair(results[pair1]['embeds'], results[pair1]['atts'], results[pair2]['embeds'], results[pair2]['atts'])
        with torch.no_grad():
            bs = results[pair1]['embeds'].size(0) #current Batch
            weights_i2t = F.softmax(sim_dict[f'{pair1}2{pair2}'][:, :bs], dim=1)
            weights_t2i = F.softmax(sim_dict[f'{pair2}2{pair1}'][:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        prop_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item() #stochastic sampling (torch.argmax for deterministic sampling)
            prop_embeds_neg.append(results[pair1]['embeds'][neg_idx]) #negative sample "slice" (1, L, width) from prop_embeds = (B, L, width=768)
        prop_embeds_neg = torch.stack(prop_embeds_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(results[pair2]['embeds'][neg_idx])
            text_atts_neg.append(results[pair2]['atts'][neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([results[pair2]['embeds'], text_embeds_neg], dim=0)
        text_atts_all = torch.cat([results[pair2]['atts'], text_atts_neg], dim=0)

        prop_embeds_all = torch.cat([prop_embeds_neg, results[pair1]['embeds']], dim=0)
        prop_atts_all = torch.cat([results[pair1]['atts'], results[pair1]['atts']], dim=0)

        pos_neg = self.cross_attention_pair(prop_embeds_all, prop_atts_all, text_embeds_all, text_atts_all)
        vl_embeddings = torch.cat([pos_pos, pos_neg], dim=0)
        # binary classification for pair (2*B, 2) itm_head=MLP head (nn.Linear)
        vl_output = self.itm_head(vl_embeddings)
        devices = results[pair1]['atts'].device
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(devices)
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        return loss_itm

    def mlm_prediction(self, text_feature, conditional, alpha):
        input_ids, labels, text_attention_mask = text_feature
        prop_embeds, prop_atts, prop_embeds_m = conditional
        # ================= MLM: Masked Language Modeling + teacher distillation ================= # (Next Word Prediction)
        # Auto-regressive text prediction

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
        loss_mlm = loss_fct(mlm_output.permute((0, 2, 1)), labels) #tensor transpose

        loss_distill_text = -torch.sum(F.log_softmax(mlm_output, dim=-1) * F.softmax(logits_m, dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()

        return (1 - alpha) * loss_mlm + alpha * loss_distill_text

    def mpm_prediction(self, text_feature, prop_feature, encoder, mtr_head_model):
        # properties = cls + masked prop sequence, is_decoder: autoregressive prediction (use only previous tokens)
        properties, prop_atts, mpm_mask, target = prop_feature
        text_embeds, text_attention_mask = text_feature

        prop_embeds_causal = encoder(inputs_embeds=properties, is_decoder=True, return_dict=True).last_hidden_state
        # Encoder-style, Recovery masked prop (non-causal, random)
        prop_output = self.text_encoder.bert(encoder_embeds=prop_embeds_causal, #input: prop
                                             attention_mask=prop_atts,
                                             encoder_hidden_states=text_embeds, #condition: text (for cross attention)
                                             encoder_attention_mask=text_attention_mask,
                                             return_dict=True,
                                             is_decoder=True,#1.causal mask 2. **cross-attention -> but real, non-causal, bidirectional masked pred
                                             mode='fusion',
                                             ).last_hidden_state[:, :-1, :]

        pred = mtr_head_model(prop_output).squeeze()
        lossfn = nn.MSELoss()
        # idx slicing from masking (above : mpm_mask = 1 for mask, 0 for keep, 0.5=masking ratio (B, L))
        return lossfn(pred[(1 - mpm_mask).to(bool)], target[(1 - mpm_mask).to(bool)])

    def cross_attention_pair(self, q_embeds, q_atts, k_embeds, k_atts):
        # all modality integrated text_encoder.bert > tri-modal cross attention
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

    def contrastive_loss(self, q_feats, k_feats=None, alpha=0.5, temp=0.7, mode='intra'):
        # from momentum model 
        q_feat, q_feat_m, q_feat_all = q_feats
        if mode == 'inter':
            k_feat, k_feat_m, k_feat_all = k_feats
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

        if mode == 'intra':
            with torch.no_grad():
                sim_q2q_m = q_feat_m @ q_feat_all / temp  # (B, B+Bq)
                sim_targets = torch.zeros(sim_q2q_m.size(), device=q_feat.device)
                sim_targets.fill_diagonal_(1)
                sim_q2q_targets = alpha * F.softmax(sim_q2q_m, dim=1) + (1 - alpha) * sim_targets
        
            sim_q2q = q_feat @ q_feat_all / temp
            loss_q2q = -torch.sum(F.log_softmax(sim_q2q, dim=1) * sim_q2q_targets, dim=1).mean()
            return sim_q2q, loss_q2q

    def set_eval_mode(self):
        # for debugging
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.prop_encoder.eval()
        self.prop_mtr_head.eval()
        self.prop_proj.eval()
        self.text_encoder.eval()
        self.text_mtr_head.eval()
        self.text_proj.eval()
        self.dist_encoder.eval()
        self.dist_mtr_head.eval()
        self.dist_proj.eval()
        self.prop_embed.eval()
        self.dist_embed_layer.eval()
        self.itm_head.eval()
        self.prop_encoder_m.eval()
        self.prop_proj_m.eval()
        self.text_encoder_m.eval()
        self.text_proj_m.eval()
        self.dist_encoder_m.eval()
        self.dist_proj_m.eval()

    def init_modal(self,name,config,hidden_width, embed_dim, norm_eps=1e-12, is_momentum=False):
        if is_momentum:
            encoder_m, proj_m = create_unimodal_models(
            config, hidden_width, embed_dim, norm_eps, is_momentum=True
            )

            setattr(self, f"{name}_encoder_m", encoder_m)
            setattr(self, f"{name}_proj_m", proj_m )
#            setattr(self, f"{name}2two_proj_m", pd_proj_m )
        else:
            encoder, mtr_head, cls_token, mask_token,  proj_layer = create_unimodal_models(
            config, hidden_width, embed_dim, norm_eps, is_momentum
            )
    
            setattr(self, f"{name}_encoder", encoder)
            setattr(self, f"{name}_mtr_head", mtr_head)
            setattr(self, f"{name}_cls", cls_token)
            setattr(self, f"{name}_mask", mask_token)
            setattr(self, f"{name}_proj", proj_layer)
#            setattr(self, f"{name}2two_proj", pd_proj)

    def _init_queue_buffer(self, prefix:str, embed_dim:int, queue_size:int):
        name= f'{prefix}_queue'
        self.register_buffer(name, torch.randn(embed_dim, queue_size) )
        setattr(self, name, nn.functional.normalize(getattr(self, name), dim=0))

    def build_config(self, **kwargs):
        if kwargs['is_momentum']== False: #student model
            return {
                "prop": {
                        "model": kwargs["prop_encoder"],
                        "proj": kwargs["prop_proj"],
                        "queue": kwargs["prop_queue"],
                        "mtr_head": kwargs["prop_mtr"],
                        "inputs": {"inputs_embeds": kwargs["prop"]["inputs_embeds"]},
                        "is_momentum": kwargs['is_momentum']
                    },
                "dist": {
                        "model": kwargs["dist_encoder"],
                        "proj": kwargs["dist_proj"],
                        "queue": kwargs["dist_queue"],
                        "mtr_head": kwargs["dist_mtr"],
                        "inputs": {"inputs_embeds": kwargs["dist"]["inputs_embeds"]},
                        "is_momentum": kwargs['is_momentum']
                    },
                "text": {
                        "model": kwargs["text_encoder"],
                        "proj": kwargs["text_proj"],
                        "queue": kwargs["text_queue"],
                        "inputs": {"input_ids": kwargs['text']['input_ids'],
                                   "attention_mask": kwargs['text']['attention_mask'],
                                   "mode": kwargs['text']['mode']
                                  },
                        "is_momentum": kwargs['is_momentum']
                    },
                   }
        else:#is_momentum=True, teacher model
            return {
                "prop": {
                        "model": kwargs["prop_encoder_m"],
                        "proj": kwargs["prop_proj_m"],
                        },
                "dist": {
                        "model": kwargs["dist_encoder_m"],
                        "proj": kwargs["dist_proj_m"],
                    },
                "text": {
                        "model": kwargs["text_encoder_m"],
                        "proj": kwargs["text_proj_m"],
                    },
                }

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

class DistEmbedLayer(torch.nn.Module):
    """
    Transforms scalar distance values into a Gaussian kernel-based distance  matrix.
    """
    def __init__(self, n_feat, start, end): #embedding_dim, start and end for distance range
        super().__init__()
        # center, exponent for trainable parameters #grid spacing like as np.linspace(start, end, n_feat)
        self.center  = nn.Parameter(start + (end - start) / (n_feat - 1) * torch.arange(n_feat) ) # (n_feat)
        self.exponent = nn.Parameter(2*torch.ones(n_feat)* n_feat/ (end-start) ) # exponent = alpha, Gaussian = exp(-alpha(d-dj)2)

    def forward(self, dist) -> torch.Tensor:
        # (B, max_pair) -> (B, max_pair, n_feat) for 3D-broadcasting
        if 1!=dist.size()[-1]:             # (B, max_pair)
            dist = dist.unsqueeze(-1)      # (B, max_pair, 1) 2D to expand 3D
        #self.center (n_feat) -> center(1,1,n_feat)
        center = self.center.reshape( tuple([ 1 for i in range(len(dist.size())-1 )]+[-1]) ) #[1,1] + [-1] = [1,1,-1]
        exponent = self.exponent.reshape( tuple([ 1 for i in range(len(dist.size())-1 )]+[-1]) ) 
        dist_f = torch.exp( -torch.abs(exponent) * torch.pow( dist - center, 2) ) #(B, max_pair, n_feat)
        # (B, ) -> (B, n_feat) for 2D-broadcasting
#        center  = self.center.reshape(1, -1) # (n_feat) > (1, n_feat)
#        exponent = self.exponent.reshape(1, -1) # (n_feat) > (1, n_feat)
#        dist_f = torch.exp( -torch.abs(exponent) * torch.pow (dist - center, 2) ) #(B, n_feat)
        return dist_f

class Embed3DStruct(torch.nn.Module):
    def __init__(self, symbol_pair_embed_length, distance_embed_length):
        super().__init__()
        self.pair_embed_length = symbol_pair_embed_length + distance_embed_length
        self.dist_embed = DistEmbedLayer(distance_embed_length, 0.0, 6.0)
        self.vocab_size = pickle.load(open('new_atom_pair_vocab.pkl', 'rb'))
        self.pair_symbol_embed = nn.Embedding( len(self.vocab_size), symbol_pair_embed_length ) #(vocab_size, embed_length)
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, self.pair_embed_length))

    def forward(self, pair_symbols, distances):
        pair_symbol_feat = self.pair_symbol_embed(pair_symbols)  # (B, N_maxpair, embed_length)
        dist_feat = self.dist_embed(distances)
        embed_feat = torch.cat([ pair_symbol_feat, dist_feat], -1)

        return embed_feat
