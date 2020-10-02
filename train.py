from modules.entities import EntityTracker
from modules.bow import BoW_encoder
from  modules.lstm_net import LSTM_net
from  modules.lstm_net2 import LSTM_net2
from  modules.lstm_net3 import LSTM_net3
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker
from modules.data_utils import Data
import modules.util as util
import tensorflow as tf
from nltk.translate import bleu_score

import numpy as np
from modules.attention import Attention
import sys


class Trainer():

    def __init__(self):

        et = EntityTracker()
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed()
        at = ActionTracker(et)
        #获得用户说的话和对应的id，获得段落对话的行数：[('good morning', 3), ("i'd like to book a table with italian food", 7), ('<SILENCE>', 13), ('in paris', 6), ('for six people please', 14)...[{'start': 0, 'end': 21}, {'start': 21, 'end': 40}, {'start': 40, 'end': 59}, {'start': 59, 'end': 72},...
        self.dataset, dialog_indices = Data(et, at).trainset
        self.dialog_indices_tr = dialog_indices[:800]
        self.dialog_indices_dev = dialog_indices[800:1000]

        self.smooth_fun = bleu_score.SmoothingFunction()
        #dim=300,vocab_size=len(self.vocab),num_features=4    =>300+88+4=392
        obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features

        #获得对话模板
        self.action_templates = at.get_action_templates()
        #对话模板长度=16
        action_size = at.action_size
        nb_hidden = 128
        #自己加的
        inputa=[]
        inputb=[]
        labela=[]
        labelb=[]
        for i in range(600):
            start,end= dialog_indices[i]['start'],dialog_indices[i]['end']
            tempa,tempb=[],[]
            for m,n in self.dataset[start:end]:
                tempa.append(m)
                tempb.append(n)
            inputa.append(tempa)
            labela.append(tempb)
        for i in range(200):
            start,end= dialog_indices[i+600]['start'],dialog_indices[i+600]['end']
            tempa,tempb=[],[]
            for m,n in self.dataset[start:end]:
                tempa.append(m)
                tempb.append(n)
            inputb.append(tempa)
            labelb.append(tempb)
        input_tenser={'inputa':inputa,'labela':labela,'inputb':inputb,'labelb':labelb}
        self.input_tenser = input_tenser
        self.graph1 = tf.Graph()
        self.graph2 = tf.Graph()

         # Session1
        # Session2
        #392,16,128
        tf.reset_default_graph()
        self.obs_size =obs_size
        self.action_size=action_size
        self.nb_hidden=nb_hidden

        self.net1 = LSTM_net(obs_size=obs_size,
                           action_size=action_size,
                           nb_hidden=nb_hidden,
                            graph = self.graph1)
        self.net2 = LSTM_net2(obs_size=obs_size,
                             action_size=action_size,
                             nb_hidden=nb_hidden,
                              graph=self.graph2)

    def train(self):

        print('\n:: training started\n')
        epochs = 1
        self.inputa = self.input_tenser['inputa']
        self.inputb = self.input_tenser['inputb']
        self.labela = self.input_tenser['labela']
        self.labelb = self.input_tenser['labelb']
        for _ in range(10000):
            loss_listb = []
            # with tf.Session(graph=self.graph1) as sess1:
            for i in range(len(self.inputa)):
                #针对于段落的实体追踪，i是每一段话
                # create entity tracker哦
                et = EntityTracker()
                # create action tracker
                at = ActionTracker(et)
                # reset network
                self.net1.reset_state()
                for j in range(1,len(self.inputa[i])):
                    #j是每一句话
                    u = self.inputa[i][j]
                    # u是用户说的话，r是位置
                    u_ent = et.extract_entities(u,self.inputa[i][0][0])
                    # u_ent_features：对et进行编码，返回4维矩阵
                    u_ent_features = et.context_features()
                    # u_emb：对用户说的话进行编码，得到一个300维度的矩阵
                    u_emb = self.emb.encode(u)
                    # 返回矩阵，每句话对应单词的位置为1，其他为0
                    u_bow = self.bow_enc.encode(u)
                        # concat features
                        # features:三个矩阵首尾相接，形成一个大的一维矩阵。维度是：(392，)
                    features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)

                    action_=self.labela[i][j]
                        # get action mask 16维度，根据提供的槽位独特设置的
                    action_mask = at.action_mask()
                    self.net1.noneforward(features,action_,action_mask)
            for i in range(len(self.inputb)):#len(self.inputb
                et = EntityTracker()
                # create action tracker
                at = ActionTracker(et)
                # reset network
                self.net1.reset_state()
                for j in range(1,len(self.inputb[i])):#self.inputb[i]
                    u = self.inputb[i][j]
                    # u是用户说的话，r是位置
                    u_ent = et.extract_entities(u,self.inputb[i][0][0])
                    # u_ent_features：对et进行编码，返回4维矩阵
                    u_ent_features = et.context_features()
                    # u_emb：对用户说的话进行编码，得到一个300维度的矩阵
                    u_emb = self.emb.encode(u)
                    # 返回矩阵，每句话对应单词的位置为1，其他为0
                    u_bow = self.bow_enc.encode(u)
                    # concat features
                    # features:三个矩阵首尾相接，形成一个大的一维矩阵。维度是：(392，)
                    features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                    action_ = self.labelb[i][j]
                    # get action mask 16维度，根据提供的槽位独特设置的
                    action_mask = at.action_mask()
                    loss = self.net1.nonetrain(features,action_,action_mask)
                    loss_listb.append(loss)

        # 开始算平均的损失，这里只是简单的求个平均
            for i in range(len(loss_listb)):
                if i ==0 :
                    temp = loss_listb[i]
                else:
                    temp+=loss_listb[i]
            avgloss =temp / 200.0
            # avgloss = tf.reduce_sum(loss_listb) / tf.to_float(200)

            u = self.inputb[1][1]
            et = EntityTracker()
            # create action tracker
            at = ActionTracker(et)
            # reset network
            self.net1.reset_state()
            # u是用户说的话，r是位置
            u_ent = et.extract_entities(u,self.inputb[1][0][0])
            # u_ent_features：对et进行编码，返回4维矩阵
            u_ent_features = et.context_features()
            # u_emb：对用户说的话进行编码，得到一个300维度的矩阵
            u_emb = self.emb.encode(u)
            # 返回矩阵，每句话对应单词的位置为1，其他为0
            u_bow = self.bow_enc.encode(u)
            # concat features
            # features:三个矩阵首尾相接，形成一个大的一维矩阵。维度是：(392，)
            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            action_ = self.labelb[1][1]
            # get action mask 16维度，根据提供的槽位独特设置的
            action_mask = at.action_mask()
            self.net2.noneop(features,action_,action_mask,avgloss)
            self.net2.save()
            #新建一个图作为中间变量
            self.graph1 = tf.Graph()
            self.net1 = LSTM_net3(obs_size = self.obs_size,
                                 action_size = self.action_size,
                                 nb_hidden = self.nb_hidden,
                                 graph = self.graph1)
            accuracy = self.evaluate()
            temp = self.getbleu()
            print(temp)
            print('准确率是')
            print(accuracy)


            # temp_loss = self.net1.construct_modell(self.input_tenser)



            # iterate through dialogs


        self.net2.save()
        print('函数')

    def dialog_train(self, dialog):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        loss = 0.
        # iterate through dialog
        for (u,r) in dialog:
            #u是用户说的话，r是位置
            u_ent = et.extract_entities(u)
            #u_ent_features：对et进行编码，返回4维矩阵
            u_ent_features = et.context_features()
            #u_emb：对用户说的话进行编码，得到一个300维度的矩阵
            u_emb = self.emb.encode(u)
            #返回矩阵，每句话对应单词的位置为1，其他为0
            u_bow = self.bow_enc.encode(u)
            # concat features
            #features:三个矩阵首尾相接，形成一个大的一维矩阵。维度是：(392，)
            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)

            # get action mask 16维度，根据提供的槽位独特设置的
            action_mask = at.action_mask()
            # forward propagation
            #  train step
            loss += self.net.train_step(features, r, action_mask)
        return loss/len(dialog)

    def evaluate(self):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network

        dialog_accuracy = 0.
        for dialog_idx in self.dialog_indices_dev:

            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)

            # create entity tracker
            et = EntityTracker()
            # create action tracker
            at = ActionTracker(et)
            # reset network
            self.net2.reset_state()

            # iterate through dialog
            correct_examples = 0

            for i in range(1,len(dialog)):
                # encode utterance
                u_ent = et.extract_entities(dialog[i][0],dialog[0][0][0])
                u_ent_features = et.context_features()
                u_emb = self.emb.encode(dialog[i][0])
                u_bow = self.bow_enc.encode(dialog[i][0])
                # concat features
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                # get action mask
                action_mask = at.action_mask()
                # forward propagation
                #  train step
                prediction = self.net2.forward(features, action_mask)
                correct_examples += int(prediction == dialog[i][1])
            # get dialog accuracy
            dialog_accuracy += correct_examples/len(dialog)

            # for (u,r) in dialog:
            #     # encode utterance
            #     print(dialog)
            #     print(u)
            #     u_ent = et.extract_entities(u)
            #     u_ent_features = et.context_features()
            #     u_emb = self.emb.encode(u)
            #     u_bow = self.bow_enc.encode(u)
            #     # concat features
            #     features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            #     # get action mask
            #     action_mask = at.action_mask()
            #     # forward propagation
            #     #  train step
            #     prediction = self.net2.forward(features, action_mask)
            #     correct_examples += int(prediction == r)
            # # get dialog accuracy
            # dialog_accuracy += correct_examples/len(dialog)

        return dialog_accuracy/num_dev_examples

    def getbleu(self):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net2.reset_state()
        dialog_accuracy = 0.
        for dialog_idx in self.dialog_indices_dev:

            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)

            # create entity tracker
            et = EntityTracker()
            # create action tracker
            at = ActionTracker(et)
            # reset network
            self.net2.reset_state()

            # iterate through dialog
            correct_examples = 0
            num = 0
            for i in range(1,len(dialog)):
                # encode utterance
                u_ent = et.extract_entities(dialog[i][0],dialog[0][0][0])
                u_ent_features = et.context_features()
                u_emb = self.emb.encode(dialog[i][0])
                u_bow = self.bow_enc.encode(dialog[i][0])
                # concat features
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                # get action mask
                action_mask = at.action_mask()
                # forward propagation
                #  train step
                prediction = self.net2.forward(features, action_mask)
                return 1

                if num % 30 == 0:
                    references = at.action_templates[prediction]
                    hypothesis = [at.action_templates[r]]
                    print('oooo')
                    print(references)
                    print(hypothesis)
                    bleu_score.sentence_bleu(references, hypothesis, weights=(0.5, 0.5),
                                             smoothing_function=self.smooth_fun.method2)
                    print('bleu的分数是：',end='')
                    print(bleu_score)
            # get dialog accuracy



if __name__ == '__main__':
    # setup trainer
    trainer = Trainer()
    # start training
    trainer.train()
