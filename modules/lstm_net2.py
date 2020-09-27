import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav
from modules.entities import EntityTracker
from modules.actions import ActionTracker
import numpy as np

class LSTM_net2():

    def __init__(self, obs_size, graph,nb_hidden=128, action_size=16):
        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.action_size = action_size
        self.graph=graph
        def __graph__():
            with graph.as_default():

                # entry points
                features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')

                #init_state_c_:Tensor("Placeholder:0", shape=(1, 128), dtype=float32)  init_state_h_:Tensor("Placeholder_1:0", shape=(1, 128), dtype=float32)
                init_state_c_, init_state_h_ = ( tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2) )
                #action_:Tensor("ground_truth_action:0", dtype=int32)
                action_ = tf.placeholder(tf.int32, name='ground_truth_action')
                #action_mask_:Tensor("action_mask:0", shape=(16,), dtype=float32)
                action_mask_ = tf.placeholder(tf.float32, [action_size], name='action_mask')
                flag_ = tf.placeholder(tf.int32, [1], name='flag')

                # input projection
                Wi = tf.get_variable('Wi', [obs_size, nb_hidden],
                        initializer=xav())
                bi = tf.get_variable('bi', [nb_hidden],
                        initializer=tf.constant_initializer(0.))

                # add relu/tanh here if necessary
                projected_features = tf.matmul(features_, Wi) + bi

                lstm_f = tf.contrib.rnn.LSTMCell(nb_hidden, state_is_tuple=True)

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
                saver = tf.train.Saver()

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
            return saver
        # build graph
        self.saver = __graph__()
        # start a session; attach to self
        with graph.as_default():
            sess = tf.Session(graph=graph)
            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            sess.run(tf.global_variables_initializer())
            self.sess = sess
        # set init state to zeros
        self.init_state_c = np.zeros([1, self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1, self.nb_hidden], dtype=np.float32)

    def getwi(self,temp):
        return tf.get_variable('Wi', [self.obs_size, self.nb_hidden])

    def getwo(self, temp):
        return tf.get_variable('Wo', [self.obs_size, self.nb_hidden])

    def getbi(self, temp):
        return tf.get_variable('bi', [self.obs_size, self.nb_hidden])

    def getbo(self, temp):
        return tf.get_variable('bo', [self.obs_size, self.nb_hidden])

            # 指定tf.map_fn返回值类型
    def noneop(self,features, action, action_mask,avgloss):
        loss , op, state_c, state_h = self.sess.run([self.loss,self.train_op, self.state.c, self.state.h],
                                                            feed_dict={
                                                                self.features_: features.reshape([1, self.obs_size]),
                                                                self.action_: [action],
                                                                self.init_state_c_: self.init_state_c,
                                                                self.init_state_h_: self.init_state_h,
                                                                self.action_mask_: action_mask,
                                                                self.loss :avgloss
                                                            })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h


        return loss


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
        self.saver.save(self.sess, 'ckpt/hcn.ckpt', global_step=0)
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
