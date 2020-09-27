import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav
from modules.entities import EntityTracker
from modules.actions import ActionTracker
import numpy as np
from modules.embed import UtteranceEmbed
from modules.bow import BoW_encoder
from modules.attention import Attention

class LSTM_net3():
    def __init__(self, obs_size, graph, nb_hidden=128, action_size=16):
        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.action_size = action_size
        self.graph=graph
        def __graph__():

            with graph.as_default():

                # tf.reset_default_graph()
                # entry points
                features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')
                #init_state_c_:Tensor("Placeholder:0", shape=(1, 128), dtype=float32)  init_state_h_:Tensor("Placeholder_1:0", shape=(1, 128), dtype=float32)
                init_state_c_, init_state_h_ = ( tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2) )
                #action_:Tensor("ground_truth_action:0", dtype=int32)
                action_ = tf.placeholder(tf.int32, name='ground_truth_action')
                #action_mask_:Tensor("action_mask:0", shape=(16,), dtype=float32)
                action_mask_ = tf.placeholder(tf.float32, [action_size], name='action_mask')
                # input projection
                Wi = tf.get_variable('Wi', [obs_size, nb_hidden],
                        initializer=xav())
                bi = tf.get_variable('bi', [nb_hidden],
                        initializer=tf.constant_initializer(0.))
                # add relu/tanh here if necessary
                projected_features = tf.matmul(features_, Wi) + bi

                # projected_features = tf.expand_dims(projected_features, dim=0)
                # num_units = projected_features.get_shape[-1]
                # sl = projected_features.get_shape[-2]
                # Attention_init = Attention()
                # projected_features_temp = Attention_init.attention_net(projected_features, sl, projected_features, num_units,
                #                                      8, 1, dropout_rate=0.5,
                #                                      is_training=True, reuse=None)

                lstm_f = tf.contrib.rnn.LSTMCell(nb_hidden, state_is_tuple=True)

                #lstm_op, state = lstm_f(inputs=projected_features_temp, state=(init_state_c_, init_state_h_))
                lstm_op, state = lstm_f(inputs=projected_features, state=(init_state_c_, init_state_h_))

                # reshape LSTM's state tuple (2,128) -> (1,256)
                state_reshaped = tf.concat(axis=1, values=(state.c, state.h))

                # output projection
                Wo = tf.get_variable('Wo', [2*nb_hidden, action_size],
                        initializer=xav())

                bo = tf.get_variable('bo', [action_size],
                        initializer=tf.constant_initializer(0.))
                # get logits
                logits = tf.matmul(state_reshaped, Wo) + bo
                # probabilities
                #  normalization : elemwise multiply with action mask
                probs = tf.multiply(tf.squeeze(tf.nn.softmax(logits)), action_mask_)

                # prediction
                prediction = tf.arg_max(probs, dimension=0)

                # loss
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_)

                # train op

                train_op = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

            # attach symbols to self
                self.loss = loss
                self.prediction = prediction
                self.probs = probs
                self.logits = logits
                self.state = state
                self.train_op = train_op

                # attach placeholders
                self.features_ = features_
                self.init_state_c_ = init_state_c_
                self.init_state_h_ = init_state_h_
                self.action_ = action_
                self.action_mask_ = action_mask_



        # build graph
        __graph__()
        with graph.as_default():
            sess = tf.Session(graph=graph)
            ckpt_state = tf.train.get_checkpoint_state('ckpt')
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            self.sess = sess
        # start a session; attach to self
        # with tf.Session(graph=graph) as sess1:
        #     global_vars = tf.global_variables()
        #     is_not_initialized = sess1.run([tf.is_variable_initialized(var) for var in global_vars])
        #     not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        #     print(not_initialized_vars)
        #     # sess1.run(tf.global_variables_initializer())
        #     # self.sess = sess1

        # global_vars = tf.global_variables()
        # is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        # not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        # print(not_initialized_vars)
        # print(len(not_initialized_vars))
        # sess.run(tf.global_variables_initializer())
        # self.sess = sess
        # sess = tf.Session(graph=graph)
        # self.sess = sess
        # with sess.as_default():
        #     tf.global_variables_initializer().run(session =self.sess)




        # set init state to zeros
        self.init_state_c = np.zeros([1, self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1, self.nb_hidden], dtype=np.float32)




    def construct_modell(self, input_tensors=None, prefix='metatrain_'):

        with self.graph.as_default():
            self.emb=UtteranceEmbed()
            self.bow_enc = BoW_encoder()

            if input_tensors is None:
                #tf.placeholder(tf.float32):一维数据
                self.inputa = tf.placeholder(tf.float32)
                self.inputb = tf.placeholder(tf.float32)
                self.labela = tf.placeholder(tf.float32)
                self.labelb = tf.placeholder(tf.float32)
            else:
                self.inputa = input_tensors['inputa']
                self.inputb = input_tensors['inputb']
                self.labela = input_tensors['labela']
                self.labelb = input_tensors['labelb']
            #管理变量作用域
            # 判断模型权重是否已存在，未存在则初始化
            loss_listb = []


                # if 'weights' in dir(self):
                #     #返回当前模块的属性列表
                #     training_scope.reuse_variables()
                #     weights = self.weights
                # else:
                #     # Define the weights
                #     self.weights = weights = self.construct_weights()

                # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            num_updates = 5
            #数据的处理
            with tf.Session(graph=self.graph) as sess1:
                for i in range(len(self.inputa)):
                    for j in range(len(self.inputa[i])):
                        u = self.inputa[i][j]
                        et = EntityTracker()
                            # create action tracker
                        at = ActionTracker(et)
                            # reset network
                        self.reset_state()
                        # u是用户说的话，r是位置
                        u_ent = et.extract_entities(u)
                        # u_ent_features：对et进行编码，返回4维矩阵
                        u_ent_features = et.context_features()
                        # u_emb：对用户说的话进行编码，得到一个300维度的矩阵
                        u_emb = self.emb.encode(u)
                        # 返回矩阵，每句话对应单词的位置为1，其他为0
                        u_bow = self.bow_enc.encode(u)
                            # concat features
                            # features:三个矩阵首尾相接，形成一个大的一维矩阵。维度是：(392，)
                        features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                        features = features.reshape([1, self.obs_size])
                        action_=self.labela[i][j]
                            # get action mask 16维度，根据提供的槽位独特设置的
                        action_mask = at.action_mask()
                        # add relu/tanh here if necessary

                        _,temp_statec,temp_stateh =  sess1.run([self.train_op,self.state.c, self.state.h], feed_dict = {
                    self.features_ : features,
                    self.action_ : [action_],
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_mask_ : action_mask
                    })
                        self.init_state_c = temp_statec
                        self.init_state_h = temp_stateh
                for i in range(len(self.inputb)):
                    for j in range(len(self.inputb[i])):
                        u = self.inputb[i][j]
                        et = EntityTracker()
                        # create action tracker
                        at = ActionTracker(et)
                        # reset network
                        self.reset_state()
                        # u是用户说的话，r是位置
                        u_ent = et.extract_entities(u)
                        # u_ent_features：对et进行编码，返回4维矩阵
                        u_ent_features = et.context_features()
                        # u_emb：对用户说的话进行编码，得到一个300维度的矩阵
                        u_emb = self.emb.encode(u)
                        # 返回矩阵，每句话对应单词的位置为1，其他为0
                        u_bow = self.bow_enc.encode(u)
                        # concat features
                        # features:三个矩阵首尾相接，形成一个大的一维矩阵。维度是：(392，)
                        features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                        features = features.reshape([1, self.obs_size])
                        action_ = self.labelb[i][j]
                        # get action mask 16维度，根据提供的槽位独特设置的
                        action_mask = at.action_mask()
                        loss = sess1.run(self.loss, feed_dict={
                            self.features_: features,
                            self.action_: [action_],
                            self.init_state_c_: self.init_state_c,
                            self.init_state_h_: self.init_state_h,
                            self.action_mask_: action_mask
                        })

                        loss_listb.append(loss)
            #开始算平均的损失，这里只是简单的求个平均
            avgloss = tf.reduce_sum(loss_listb) / 200.0

            return avgloss




            # 指定tf.map_fn返回值类型
    def noneforward(self,features,r,action_mask):

        probs, op, state_c, state_h = self.sess.run([self.probs, self.train_op, self.state.c, self.state.h],
                                                                feed_dict={
                                                                    self.features_: features.reshape([1, self.obs_size]),
                                                                    self.action_: [r],
                                                                    self.init_state_c_: self.init_state_c,
                                                                    self.init_state_h_: self.init_state_h,
                                                                    self.action_mask_: action_mask

                                                                })
            # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
            # return argmax
    def changeWi(self,temp):
        tf.get_variable_scope().reuse_variables()
        self.sess.run(tf.assign(tf.get_variable('Wi', [self.obs_size, self.nb_hidden]), temp))
    def changeWo(self,temp):
        tf.get_variable_scope().reuse_variables()
        self.sess.run(tf.assign(tf.get_variable('Wo', [self.obs_size, self.nb_hidden]), temp))
    def changebi(self,temp):
        tf.get_variable_scope().reuse_variables()
        self.sess.run(tf.assign(tf.get_variable('bi', [self.obs_size, self.nb_hidden]), temp))
    def changebo(self,temp):
        tf.get_variable_scope().reuse_variables()
        self.sess.run(tf.assign(tf.get_variable('bo', [self.obs_size, self.nb_hidden]), temp))
    def nonetrain(self, features, action, action_mask):
        loss_value, state_c, state_h = self.sess.run( [ self.loss, self.state.c, self.state.h],
                feed_dict = {
                    self.features_ : features.reshape([1, self.obs_size]),
                    self.action_ : [action],
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_mask_ : action_mask
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        return loss_value




    # forward propagation
    def forward(self, features, action_mask):
        # forward
        probs, prediction, state_c, state_h = self.sess.run( [self.probs, self.prediction, self.state.c, self.state.h], 
                feed_dict = { 
                    self.features_ : features.reshape([1,self.obs_size]), 
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_mask_ : action_mask
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        # return argmax
        return prediction

    # training
    def train_step(self, features, action, action_mask):
        _, loss_value, state_c, state_h = self.sess.run( [self.train_op, self.loss, self.state.c, self.state.h],
                feed_dict = {
                    self.features_ : features.reshape([1, self.obs_size]),
                    self.action_ : [action],
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_mask_ : action_mask
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        return loss_value

    def reset_state(self):
        # set init state to zeros
        self.init_state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)

    # save session to checkpoint
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'ckpt/hcn.ckpt', global_step=0)
        print('\n:: saved to ckpt/hcn.ckpt \n')

    # restore session from checkpoint
    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('ckpt/')
        if ckpt and ckpt.model_checkpoint_path:
            print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('\n:: <ERR> checkpoint not found! \n')
