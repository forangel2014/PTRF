# Copyright 2021 Tsinghua SPMI lab, Author: Sun Wangtao swt17@tsinghua.org.cn


"""Pre-trains an Energy-Based TRF model."""

import argparse
import os
import re
import logging
import numpy as np
import random
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import time

#from metric import evaluate


class Config():
    def __init__(self,
                model_dir,
                model_load,
                vocab_size = 16296,
                dataset_name = 'librispeech_3layer',
                train_method = 'dnce',
                task = 'tail enetity prediction',
                n = 1,
                k = 1,
                max_l = 1000,
                alpha = 0,
                valid_num = 10,
                lr = 1e-7,
                weight_decay = 0,
                batchsize = 10,
                epoch = 10,
                episilon = 1e-20):
        self.model_dir = model_dir
        self.model_load = model_load
        self.vocab_size = vocab_size
        self.dataset_name = dataset_name
        self.train_method = train_method
        self.task = task
        self.n = n
        self.k = k
        self.max_l = max_l
        self.alpha = alpha
        self.valid_num = valid_num
        self.lr = lr
        self.weight_decay = weight_decay
        self.batchsize = batchsize
        self.epoch = epoch
        self.episilon = episilon

class EnergyBasedKGModel(torch.nn.Module):
    def __init__(self, config: Config):
        super(EnergyBasedKGModel, self).__init__()
        self.is_trained = False
        self.config = config

        self.pi_gpt2 = torch.nn.Parameter(torch.zeros(config.max_l), requires_grad=False)
        self.pi_bert = torch.nn.Parameter(torch.zeros(config.max_l), requires_grad=False)
        #修改初始化
        self.zeta = torch.nn.Parameter(np.log(config.vocab_size)*torch.tensor(range(config.max_l))/100)
        self.random_sample = []
        #generator
        self.generator_name = "gpt2"
        self.generator_path = "/home/sunwangtao/EBKG/swbd_exp/generator_file/"
        self.generator_config = GPT2Config.from_pretrained(self.generator_path)
        self.generator_tokenizer = GPT2Tokenizer.from_pretrained(self.generator_path)
        self.generator_model = GPT2LMHeadModel.from_pretrained(self.generator_path)
        
        #discriminator
        self.discriminator_name = "bert-base-uncased"
        self.discriminator_path = "/home/sunwangtao/EBKG/swbd_exp/discriminator_file/"
        self.discriminator_config = BertConfig.from_pretrained(self.discriminator_path)
        self.discriminator_tokenizer = BertTokenizer.from_pretrained(self.discriminator_path)
        #self.discriminator_model = BertModel.from_pretrained(self.discriminator_path)
        self.discriminator_model = BertModel(config=self.discriminator_config)
        

        #self.discriminator_special_tokens_dict = {'bos_token': '<|endoftext|>'}
        #self.discriminator_tokenizer.add_special_tokens(self.discriminator_special_tokens_dict)
        #self.discriminator_model.resize_token_embeddings(len(self.discriminator_tokenizer))
        self.discriminator_linear1 = torch.nn.Linear(self.discriminator_config.hidden_size, 1)
        self.discriminator_linear2 = torch.nn.Linear(self.discriminator_config.hidden_size, 1)
        self.discriminator_linear3 = torch.nn.Linear(self.discriminator_config.hidden_size, 1)
        self.train_mode()

    def train_mode(self):
        self.train()
        self.generator_model.train()
        self.discriminator_model.train()

    def eval_mode(self):
        self.eval()
        self.generator_model.eval()
        self.discriminator_model.eval()

    def generate(self, l, text='<|endoftext|>'):
        generate_text = text
        logq = 0
        indexed_tokens = self.generator_tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens]).cuda(1)
        if (l >= 0):
            for i in range(l-2):
                repr_vec = self.get_repr_vec(self.generator_model, tokens_tensor)[-1,:]
                q_distribution = F.softmax(repr_vec)
                generate_index = self.sample_prob(q_distribution)
                generate_text += self.generator_tokenizer.decode(generate_index)
                indexed_tokens += [generate_index]
                tokens_tensor = torch.tensor([indexed_tokens]).cuda(1)
                logq += torch.log(torch.clamp(q_distribution[generate_index], min=self.config.episilon).cuda(1))
            repr_vec = self.get_repr_vec(self.generator_model, tokens_tensor)[-1,:]
            q_distribution = F.softmax(repr_vec)
            logq += torch.log(torch.clamp(q_distribution[50256], min=self.config.episilon).cuda(1))
            generate_text += '<|endoftext|>'
        else:
            next_text = None
            while (next_text != "<|endoftext|>"):
                repr_vec = self.get_repr_vec(self.generator_model, tokens_tensor)[-1,:]
                q_distribution = F.softmax(repr_vec)
                #generate_index = self.sample_top_k(repr_vec)
                generate_index = self.sample_prob(q_distribution)
                next_text = self.generator_tokenizer.decode(generate_index)
                generate_text += next_text
                indexed_tokens += [generate_index]
                tokens_tensor = torch.tensor([indexed_tokens]).cuda(1)
                logq += torch.log(torch.clamp(q_distribution[generate_index], min=self.config.episilon).cuda(1))
        return generate_text, logq
    
    def generate_logq(self, input_tensor):
        logq = 0
        if (input_tensor.dim() == 2):
            input_tensor = input_tensor[0]
        mask = input_tensor.gt(0)
        input_tensor = input_tensor[mask]
        repr_vec = self.get_repr_vec(self.generator_model, input_tensor)
        for i in range(input_tensor.shape[0]-1):
            next_id = input_tensor[i+1]
            q_distribution = F.softmax(repr_vec[i])
            logq += torch.log(torch.clamp(q_distribution[next_id], min=self.config.episilon).cuda(1))
        return logq

    def get_repr_vec(self, model, x):
        model_output = model(x)[0]
        if (model_output.dim() == 3):
            model_output = model_output[0]
        repr_vec = model_output
        return repr_vec

    def sample_prob(self, q):
        return np.random.choice(len(q), p=q.cpu().detach().numpy())

    def sample_top_k(self, predictions, k=10):
        predicted_index = random.choice(predictions.sort(descending=True)[1][:k]).item()
        return predicted_index

    def discriminate(self, tensor):
        repr_vec = self.get_repr_vec(self.discriminator_model, tensor)
        E = (self.discriminator_linear1(repr_vec[0,:]) + self.discriminator_linear3(repr_vec[-1,:]) + \
            self.discriminator_linear2(torch.mean(repr_vec[1:-1,:], axis=0)))*250
        #E = self.discriminator_linear1(repr_vec[0,:])*300 #+ self.discriminator_linear3(repr_vec[-1,:]))*(-100)
        #p_hat = torch.exp(-E)
        return E

    def get_logp(self, tensor):
        l = tensor.shape[1]
        logp = -(self.discriminate(tensor) + self.zeta[l]) + torch.log(torch.clamp(self.pi_bert[l], min=self.config.episilon))
        return logp

    def forward(self, input_tensor):

        if (self.config.train_method == 'finetune_gpt2'):
            #input_tensor = self.triple2tensor(self.generator_tokenizer, triple)
            for i in range(input_tensor.shape[0]):
                logq = self.generate_logq(input_tensor[i,:])
                if i == 0:
                    loss = -logq
                else:
                    loss -= logq
            return loss

        if (self.config.train_method == 'finetune_ebm'):
            loss = torch.Tensor([0]).cuda(1)
            for i in range(input_tensor.shape[0]):
                for j in range(input_tensor.shape[1]):
                    if (input_tensor[i][j].cpu().detach().numpy() == -1):
                        original_tensor = input_tensor[i][:j]
                        break
                    if (j == input_tensor.shape[1]-1):
                        original_tensor = input_tensor[i]
                l0 = original_tensor.shape[0]
                with torch.no_grad():
                    logq = self.generate_logq(original_tensor) 
                           #+ torch.log(torch.clamp(self.pi_gpt2[l0], min=self.config.episilon))
                text = self.generator_tokenizer.decode(original_tensor)[13:-13]
                d_tensor = self.discriminator_tokenizer.encode(text, return_tensors="pt").cuda(1)
                #l = d_tensor.shape[1]
                logp = self.get_logp(d_tensor)#self.discriminate(d_tensor) + self.zeta[l]#-torch.log(torch.clamp(self.pi_bert[l], min=self.config.episilon)) +\
                loss += torch.abs(logq - logp)
            return loss

        if (self.config.train_method == 'nce'):
            n = torch.Tensor([self.config.n]).cuda(1)
            k = torch.Tensor([self.config.k]).cuda(1)
            loss_data = torch.Tensor([0]).cuda(1)
            loss_noise = torch.Tensor([0]).cuda(1)
            data_sample_num = input_tensor.shape[0]
            noise_sample_num = int(input_tensor.shape[0]/n*k)
            data_right_num = 0
            noise_right_num = 0
            for i in range(data_sample_num):
                #input_tensor = self.triple2tensor(self.generator_tokenizer, input_triple)
                for j in range(input_tensor.shape[1]):
                    if (input_tensor[i][j].cpu().detach().numpy() == -1):
                        original_tensor = input_tensor[i][:j]
                        break
                    if (j == input_tensor.shape[1]-1):
                        original_tensor = input_tensor[i]
                l0 = original_tensor.shape[0]
                with torch.no_grad():
                    logq = self.generate_logq(original_tensor)
                           #+ torch.log(torch.clamp(self.pi_gpt2[l0], min=self.config.episilon))
                text = self.generator_tokenizer.decode(original_tensor)[13:-13]
                d_tensor = self.discriminator_tokenizer.encode(text, return_tensors="pt").cuda(1)
                #l = d_tensor.shape[1]
                logp = self.get_logp(d_tensor)#-(self.discriminate(d_tensor) + self.zeta[l])#torch.log(torch.clamp(self.pi_bert[l], min=self.config.episilon).cuda(1)) - \   
                #loss = -log(n*p_hat/(n*p_hat+k*q)) = -log(1/(1+k/n*q/p_hat)) = -log(1/(1+exp(log(k/n)+log(q)-log(p_hat))))
                PC0_post = torch.sigmoid(torch.log(n)-torch.log(k)+logp-logq)
                if PC0_post.cpu().detach().numpy() > 0.5:
                    data_right_num += 1
                loss_data += -torch.log(torch.clamp(PC0_post, min=self.config.episilon).cuda(1))
            for i in range(noise_sample_num):
                l0 = self.sample_prob(self.pi_gpt2)
                with torch.no_grad():
                    generate_seq, logq = self.generate(l0)
                    #logq += torch.log(torch.clamp(self.pi_gpt2[l0], min=self.config.episilon))
                d_tensor = self.discriminator_tokenizer.encode(generate_seq[13:-13], return_tensors="pt").cuda(1)
                #l = d_tensor.shape[1]
                logp = self.get_logp(d_tensor)#-(self.discriminate(d_tensor) + self.zeta[l])#torch.log(torch.clamp(self.pi_bert[l], min=self.config.episilon).cuda(1)) - \
                #loss = -log(k*q/(n*p_hat+k*q)) = -log(1/(1+n/k*p_hat/q)) = -log(1/(1+exp(log(n/k)+log(p_hat)-log(q))))
                PC1_post = torch.sigmoid(torch.log(k)-torch.log(n)+logq-logp)
                if PC1_post.cpu().detach().numpy() > 0.5:
                    noise_right_num += 1
                loss_noise += -torch.log(torch.clamp(PC1_post, min=self.config.episilon).cuda(1))
            return loss_data[0], loss_noise[0], data_right_num, noise_right_num, data_sample_num, noise_sample_num

        if (self.config.train_method == 'dnce'):
            n = torch.Tensor([self.config.n]).cuda(1)
            k = torch.Tensor([self.config.k]).cuda(1)
            loss_data = torch.Tensor([0]).cuda(1)
            loss_noise = torch.Tensor([0]).cuda(1)
            loss_generator = torch.Tensor([0]).cuda(1)
            data_sample_num = input_tensor.shape[0]
            noise_sample_num = int(input_tensor.shape[0]/n*k)
            data_right_num = 0
            noise_right_num = 0
            for i in range(data_sample_num):
                #input_tensor = self.triple2tensor(self.generator_tokenizer, input_triple)
                for j in range(input_tensor.shape[1]):
                    if (input_tensor[i][j].cpu().detach().numpy() == -1):
                        original_tensor = input_tensor[i][:j]
                        break
                    if (j == input_tensor.shape[1]-1):
                        original_tensor = input_tensor[i]
                l = original_tensor.shape[0]
                logq = self.generate_logq(original_tensor)
                       #+ torch.log(torch.clamp(self.pi_gpt2[l], min=self.config.episilon))
                text = self.generator_tokenizer.decode(original_tensor)[13:-13]
                d_tensor = self.discriminator_tokenizer.encode(text, return_tensors="pt").cuda(1)
                logp = self.get_logp(d_tensor)#-(self.discriminate(d_tensor) + self.zeta[l])#torch.log(torch.clamp(self.pi_bert[l], min=self.config.episilon).cuda(1)) - \
                #loss = -log(n*p_hat/(n*p_hat+k*q)) = -log(1/(1+k/n*q/p_hat)) = -log(1/(1+exp(log(k/n)+log(q)-log(p_hat))))
                PC0_post = torch.sigmoid(torch.log(n)-torch.log(k)+logp-logq)
                if PC0_post.cpu().detach().numpy() > 0.5:
                    data_right_num += 1
                loss_data += -torch.log(torch.clamp(PC0_post, min=self.config.episilon).cuda(1)) #+ torch.abs(logq-logp)
                loss_generator += -logq
            for i in range(noise_sample_num):
                if i < int(noise_sample_num * self.config.alpha):
                    generate_seq = self.random_sample[i]
                    generate_tensor = self.generator_tokenizer.encode(generate_seq, return_tensors="pt").cuda(1)
                    with torch.no_grad():
                        logq = self.generate_logq(generate_tensor)
                else:
                    l0 = self.sample_prob(self.pi_gpt2)
                    with torch.no_grad():
                        generate_seq, logq = self.generate(l0)
                d_tensor = self.discriminator_tokenizer.encode(generate_seq[13:-13], return_tensors="pt").cuda(1)
                #l = d_tensor.shape[1]
                logp = self.get_logp(d_tensor)#-(self.discriminate(d_tensor) + self.zeta[l])#torch.log(torch.clamp(self.pi_bert[l], min=self.config.episilon).cuda(1)) - \
                #loss = -log(k*q/(n*p_hat+k*q)) = -log(1/(1+n/k*p_hat/q)) = -log(1/(1+exp(log(n/k)+log(p_hat)-log(q))))
                PC1_post = torch.sigmoid(torch.log(k)-torch.log(n)+logq-logp)
                if PC1_post.cpu().detach().numpy() > 0.5:
                    noise_right_num += 1
                loss_noise += -torch.log(torch.clamp(PC1_post, min=self.config.episilon).cuda(1)) #+ torch.abs(logq-logp)
            return loss_data[0], loss_noise[0], data_right_num, noise_right_num, data_sample_num, noise_sample_num, loss_generator[0]

    def stat_length(self, trainset):
        for i in range(100000):
            text = trainset[i][:-1]
            text_gpt = "<|endoftext|>"+text+".<|endoftext|>"
            seq_gpt = self.generator_tokenizer.encode(text_gpt, return_tensors="pt")
            l = seq_gpt.shape[1]
            self.pi_gpt2[l] += 1
            seq_bert = self.discriminator_tokenizer.encode(text, return_tensors="pt")
            l = seq_bert.shape[1]
            self.pi_bert[l] += 1
        self.pi_gpt2 /= torch.sum(self.pi_gpt2)
        self.pi_bert /= torch.sum(self.pi_bert)

    def get_random_sample(self, trainset):
        self.random_sample = []
        noise_sample_num = int(self.config.batchsize/self.config.n*self.config.k)
        for i in range(int(noise_sample_num * self.config.alpha)):
            index = random.randint(0,len(trainset))
            text = trainset[index][:-1]
            text_gpt = "<|endoftext|>"+text+".<|endoftext|>"
            self.random_sample.append(text_gpt)

    def call_generator_ppl(self, validset):
        self.eval_mode()
        with torch.no_grad():
            Nt = 0
            logp = torch.Tensor([0]).cuda(1)
            #random.shuffle(validset)
            for i in range(len(validset)):
                tensor = self.generator_tokenizer.encode("<|endoftext|>"+validset[i][:-1]+".<|endoftext|>", return_tensors='pt').cuda(1)
                Nt += tensor.shape[1]
                logp += self.generate_logq(tensor)
        logp = logp.cpu().detach().numpy()
        ppl = np.exp(-logp/Nt)
        self.train_mode()
        return ppl

    def call_discriminator_ppl(self, validset):
        self.eval_mode()
        with torch.no_grad():
            Nt = 0
            logp = torch.Tensor([0]).cuda(1)
            #random.shuffle(validset)
            for i in range(len(validset)):
                d_tensor = self.discriminator_tokenizer.encode(validset[i][:-1], return_tensors='pt').cuda(1)
                l = d_tensor.shape[1]
                Nt += l
                logp += self.get_logp(d_tensor)#-(self.discriminate(d_tensor) + self.zeta[l])#torch.log(torch.clamp(self.pi_bert[l], min=self.config.episilon).cuda(1)) - \
        logp = logp.cpu().detach().numpy()
        ppl = np.exp(-logp/Nt)
        self.train_mode()
        return ppl

    def call_dnce_loss(self, validset):
        self.eval_mode()
        loss = torch.Tensor([0]).cuda(1)
        n = torch.Tensor([self.config.n]).cuda(1)
        k = torch.Tensor([self.config.k]).cuda(1)
        with torch.no_grad():
            for i in range(len(validset)):
                g_tensor = self.generator_tokenizer.encode("<|endoftext|>"+validset[i][:-1]+".<|endoftext|>", return_tensors='pt').cuda(1)
                d_tensor = self.discriminator_tokenizer.encode(validset[i][:-1]+".", return_tensors='pt').cuda(1)
                #l = d_tensor.shape[1]
                logq = self.generate_logq(g_tensor)
                logp = self.get_logp(d_tensor)#-(self.discriminate(d_tensor) + self.zeta[l])
                PC0_post = torch.sigmoid(torch.log(n)-torch.log(k)+logp-logq)
                loss += -torch.log(torch.clamp(PC0_post, min=self.config.episilon).cuda(1))
        self.train_mode()
        return loss.detach().cpu().numpy()[0]/len(validset)

def early_stopping(method, train_loss_list, valid_loss_list):
    if len(valid_loss_list) == 1 or len(train_loss_list) == 1:
        return False
    if method == 'finetune_ebm':
        if len(train_loss_list) < 200:
            return False
        loss_now = sum(train_loss_list[-100:])
        loss_last = sum(train_loss_list[-200:-100])
    else:
        loss_now = valid_loss_list[-1]
        loss_last = valid_loss_list[-2]
    info = "loss_last = " + str(loss_last) + " loss_now = " + str(loss_now)
    print(info)
    logging.debug(info)
    return loss_last < loss_now*0
    
def train_and_eval(model, trainset, validset, steps):
    #model = model.module
    model.train_mode()
    optimizer = optim.Adam(
        [
        {'params': model.generator_model.parameters(), 'lr': model.config.lr * 1},
        {'params': model.discriminator_model.parameters(), 'lr': model.config.lr * 2},
        {'params': model.discriminator_linear1.parameters(), 'lr': model.config.lr * 2},
        {'params': model.discriminator_linear2.parameters(), 'lr': model.config.lr * 2},
        {'params': model.discriminator_linear3.parameters(), 'lr': model.config.lr * 2},
        {'params': model.zeta, 'lr': model.config.lr * 1000}
        ], lr=model.config.lr, weight_decay=model.config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    train_num = len(trainset)
    random.shuffle(trainset)
    if not model.config.model_load:
        model.stat_length(trainset)
    #print(model.pi_gpt2.cpu().detach().numpy())
    generator_train_flag = True
    time_start=time.time()

    for e in range(model.config.epoch):
        train_loss_list = []
        valid_loss_list = []
        for b in range(train_num // model.config.batchsize):
            optimizer.zero_grad()
            text = trainset[b*model.config.batchsize:(b+1)*model.config.batchsize]

            max_len = 0
            total_len = 0
            for i in range(model.config.batchsize):
                text[i] = text[i][:-1]
                text[i] = model.generator_tokenizer.encode("<|endoftext|>"+text[i]+".<|endoftext|>", return_tensors="pt").cuda(1)
                total_len += len(text[i][0])
                if (len(text[i][0]) > max_len):
                    max_len = len(text[i][0])
            
            input_tensor = torch.cat((text[0], (-torch.ones([1, max_len-len(text[0][0])])).type_as(text[0]).cuda(1)), 1)
            for i in range(1, model.config.batchsize):
                tensor_pad = torch.cat((text[i], (-torch.ones([1, max_len-len(text[i][0])])).type_as(text[0]).cuda(1)), 1)
                input_tensor = torch.cat((input_tensor, tensor_pad), 0)
            

            if (model.config.train_method == 'finetune_gpt2'):
                model.generator_model.train()
                model.discriminator_model.eval()
                loss = model(input_tensor).sum()
                loss_num = loss.cpu().detach().numpy()/total_len
                loss.backward()
                optimizer.step()
                info = "epoch "+str(e)+" step "+str(b)+" generator_finetune_loss = "+str(loss_num)
                print(info)
                logging.debug(info)
                train_loss_list.append(loss_num)
                writer.add_scalar('generator_finetune_loss_per_token', loss_num, b+steps)
            
            if (model.config.train_method == 'finetune_ebm'):
                model.generator_model.eval()
                model.discriminator_model.train()
                #model.eval_mode()
                loss = model(input_tensor).sum()
                loss_num = loss.cpu().detach().numpy()/total_len
                loss.backward()
                optimizer.step()
                info = "epoch "+str(e)+" step "+str(b)+" ebm_finetune_loss = "+str(loss_num)
                print(info)
                logging.debug(info)
                train_loss_list.append(loss_num)
                writer.add_scalar('ebm_finetune_loss_per_token', loss_num, b+steps)

            if (model.config.train_method == 'nce'):
                n = model.config.n
                k = model.config.k
                model.generator_model.eval()
                model.discriminator_model.train()
                #model.eval_mode()
                loss_data, loss_noise, data_right_num, noise_right_num, data_sample_num, noise_sample_num = model(input_tensor)
                loss_total = loss_data + loss_noise
                loss_total_ave = loss_total.cpu().detach().numpy() / (data_sample_num + noise_sample_num)
                loss_data_ave = loss_data.cpu().detach().numpy() / data_sample_num
                loss_noise_ave = loss_noise.cpu().detach().numpy()/ noise_sample_num
                acc_d = data_right_num / data_sample_num
                acc_n = noise_right_num / noise_sample_num
                acc_total = (data_right_num + noise_right_num) / (data_sample_num + noise_sample_num)
                loss_total.backward()
                optimizer.step()
                info = "epoch " + str(e) + " step " + str(b) + " NCE_total_loss = " + str(loss_total_ave) \
                        + " NCE_data_loss = " + str(loss_data_ave)+" NCE_noise_loss = "+str(loss_noise_ave) \
                        + " acc_d = " + str(acc_d) + " acc_n = " + str(acc_n) + " acc_total = " + str(acc_total)
                print(info)
                logging.debug(info)
                train_loss_list.append(loss_total_ave)
                writer.add_scalar('NCE_total_loss', loss_total_ave, b+steps)
                writer.add_scalar('NCE_data_loss', loss_data_ave, b+steps)
                writer.add_scalar('NCE_noise_loss', loss_noise_ave, b+steps)
                writer.add_scalar('acc_d', acc_d, b+steps)
                writer.add_scalar('acc_n', acc_n, b+steps)
                writer.add_scalar('acc_total', acc_total, b+steps)

            if (model.config.train_method == 'dnce'):
                n = model.config.n
                k = model.config.k
                model.generator_model.train()
                model.discriminator_model.train()
                model.get_random_sample(trainset)
                #model.eval_mode()
                loss_data, loss_noise, data_right_num, noise_right_num, \
                data_sample_num, noise_sample_num, loss_generator = model(input_tensor)
                loss_total = loss_data + loss_noise
                train_loss_list.append(loss_total.cpu().detach().numpy())
                loss_total_ave = loss_total.cpu().detach().numpy() / (data_sample_num + noise_sample_num)
                loss_data_ave = loss_data.cpu().detach().numpy() / data_sample_num
                loss_noise_ave = loss_noise.cpu().detach().numpy()/ noise_sample_num
                loss_generator_ave = loss_generator.cpu().detach().numpy()/total_len
                acc_d = data_right_num / data_sample_num
                acc_n = noise_right_num / noise_sample_num
                acc_total = (data_right_num + noise_right_num) / (data_sample_num + noise_sample_num)
                if generator_train_flag:
                    (loss_total+loss_generator).backward()
                else:
                    loss_total.backward()
                optimizer.step()
                info = "epoch " + str(e) + " step " + str(b) + " NCE_total_loss = " + str(loss_total_ave) \
                        + " NCE_data_loss = " + str(loss_data_ave)+" NCE_noise_loss = "+str(loss_noise_ave) \
                        + " generator_loss = " + str(loss_generator_ave)\
                        + " acc_d = " + str(acc_d) + " acc_n = " + str(acc_n) + " acc_total = " + str(acc_total)
                print(info)
                logging.debug(info)
                writer.add_scalar('NCE_total_loss', loss_total_ave, b+steps)
                writer.add_scalar('NCE_data_loss', loss_data_ave, b+steps)
                writer.add_scalar('NCE_noise_loss', loss_noise_ave, b+steps)
                writer.add_scalar('generator_loss', loss_generator, b+steps)
                writer.add_scalar('acc_d', acc_d, b+steps)
                writer.add_scalar('acc_n', acc_n, b+steps)
                writer.add_scalar('acc_total', acc_total, b+steps)

            time_end=time.time()
            cost_time = time_end-time_start
            print('totally cost = ', cost_time)
            logging.debug(cost_time)

            config = model.config
            if (b % 100 == 0):
                if config.train_method == 'dnce':
                    nce_loss = valid(model, validset)
                    ppl = model.call_discriminator_ppl(validset)[0]
                    valid_loss_list.append(ppl)
                    info = 'ppl = ' + str(ppl)
                    print(info)
                    logging.debug(info)
                    #torch.save(model.state_dict(), \
                    #    os.path.join(config.model_dir,config.dataset_name+'_'+config.train_method+'_'+str(b)+'.pt'))
                elif not config.train_method == 'finetune_ebm':
                    valid_loss = valid(model, validset)
                    valid_loss_list.append(valid_loss)
                if early_stopping(config.train_method, train_loss_list, valid_loss_list):
                    if config.train_method == 'finetune_ebm':
                        final_ppl = valid(model, validset)
                        info = 'final_ppl = ' + str(final_ppl)
                        print(info)
                        logging.debug(info)
                    if config.train_method == 'dnce':
                        config.lr *= 0.5
                        scheduler.step()
                        info = 'converge, loss = ' + str(valid_loss_list[-1]) + ' lr = ' + str(config.lr)
                        print(info)
                        logging.debug(info)
                        continue
                        
                    print("stop training")
                    logging.debug("stop training")
                    return

                torch.save(model.state_dict(), os.path.join(config.model_dir,config.dataset_name+'_'+config.train_method+'.pt'))
                
            
def valid(model, validset):
    model.eval_mode()
    if (model.config.train_method == 'finetune_gpt2'):
        valid_loss = model.call_generator_ppl(validset)[0]
    if (model.config.train_method == 'finetune_ebm'):
        valid_loss = model.call_discriminator_ppl(validset)[0]
    if (model.config.train_method == 'nce'):
        valid_loss = model.call_discriminator_ppl(validset)[0]
    if (model.config.train_method == 'dnce'):
        #valid_loss_generator = model.call_generator_ppl(validset)
        #valid_loss_ebm = model.call_discriminator_ppl(validset)
        #info = "valid_loss_generator = " + str(valid_loss_generator[0]) + " valid_loss_ebm = " + str(valid_loss_ebm[0])
        #print(info)
        #logging.debug(info)
        #return valid_loss_generator[0], valid_loss_ebm[0]
        valid_loss = model.call_dnce_loss(validset)
    model.train_mode()
    return valid_loss


def main():
    data_dir = "/mnt/workspace/swt/librispeech-lm-corpus/"
    model_dir = "/mnt/workspace/swt/model/"
    #trainset_dir = data_dir + "words.txt"
    trainset_dir = "/mnt/workspace/swt/librispeech_train_data/pytorchnn_v2/train1.txt"
    validset_dir = "/mnt/workspace/swt/librispeech_train_data/pytorchnn_v2/valid1.txt"
    with open(trainset_dir) as f:
        trainset = f.readlines()
    with open(validset_dir) as f:
        validset = f.readlines()
    random.shuffle(validset)
    validset = validset[:1000]
    testset = None
    config = Config(model_dir=model_dir, model_load=False)

    log_name = config.model_dir+config.dataset_name+'_'+config.train_method+'_loss.log'

    ebkgm = EnergyBasedKGModel(config=config)

    if config.model_load:
        ebkgm.load_state_dict(torch.load(os.path.join(config.model_dir, 'librispeech_valid_redefine_finetune_ebm.pt')))
        #ebkgm.load_state_dict(torch.load(os.path.join(config.model_dir, config.dataset_name+'_finetune_ebm.pt')))
        #ebkgm.load_state_dict(torch.load(os.path.join(config.model_dir, config.dataset_name+'_base.pt')))
        filemode = 'a'
        try:
            with open(log_name) as f:
                lines = f.readlines()
            steps = len(lines)
        except:
            steps = 0
    else:
        filemode = 'w'
        steps = 0

    logging.basicConfig(level=logging.DEBUG,
                        filename=log_name,
                        filemode=filemode,
                        format='')

    #if torch.cuda.is_available():
    #    ebkgm.cuda(1)

    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    ebkgm = torch.nn.DataParallel(ebkgm)
    #ebkgm.zeta = torch.nn.Parameter(np.log(ebkgm.config.vocab_size)*torch.tensor(range(200))/100)
    ebkgm = ebkgm.cuda(1)
    train_and_eval(ebkgm, trainset, validset, steps)
    #valid(ebkgm.module, testset)

if __name__ == "__main__":
    writer = SummaryWriter(log_dir="/home/sunwangtao/EBKG/librispeech_exp/log")
    main()