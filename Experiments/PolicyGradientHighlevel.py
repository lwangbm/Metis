"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.
Policy Gradient, Reinforcement Learning.
"""
import numpy as np
import tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 suffix=''):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.suffix = suffix

        """
        self.ep_obs, self.ep_as, self.ep_rs: observation, action, reward, recorded for each batch
        """
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]

        """
        self.tput_batch: record throughput for each batch, used to show progress while training
        self.tput_persisit, self.episode: persist to record throughput, used to be stored and plot later
        
        """
        self.tput_batch = []
        self.tput_persisit = []
        self.time_persisit = []
        self.entropy_persist = []
        self.episode = []
        # TODO self.vio = []: violation

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()



    def _build_net(self):
        with tf.name_scope('inputs_clustering' + self.suffix):
            self.tf_obs = tf.placeholder(tf.float32,[None,self.n_features],name='observation_clustering' + self.suffix)
            self.tf_acts = tf.placeholder(tf.int32,[None,],name='actions_num_clustering' + self.suffix)
            self.tf_vt = tf.placeholder(tf.float32,[None,],name='actions_value_clustering' + self.suffix)
            self.entropy_weight = tf.placeholder(tf.float32, shape=(), name='entropy_weight_clustering' + self.suffix)


        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=128,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1_clustering' + self.suffix
        )
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2_clustering' + self.suffix
        )
        self.all_act_prob = tf.nn.softmax(all_act,name='act_prob_clustering' + self.suffix)

        with tf.name_scope('loss_clustering' + self.suffix):
            neg_log_prob = tf.reduce_sum(-tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * tf.one_hot(indices=self.tf_acts,depth=self.n_actions),axis=1)

            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
            loss += self.entropy_weight * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))
            self.entropy = tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1)) #self.entropy_weight
            self.loss = loss

        with tf.name_scope('train_clustering' + self.suffix):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            # decay_rate =0.99999 # 0.999
            # learning_rate = 1e-1
            # self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

    def choose_action(self,observation):
        prob_weights = self.sess.run(self.all_act_prob,feed_dict={self.tf_obs:observation})#(4,) ->(1,4)
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action,prob_weights

    def choose_action_determine(self,observation):
        prob_weights = self.sess.run(self.all_act_prob,feed_dict={self.tf_obs:observation})#(4,) ->(1,4)
        action = np.argmax(prob_weights.ravel())
        return action,prob_weights

    def store_training_samples_per_episode(self,s,a,r):
        self.ep_obs.extend(s)
        self.ep_as.extend(a)
        self.ep_rs.extend(r)

    def store_tput_per_episode(self,tput,episode, time_duration):
        self.tput_batch.append(tput)
        self.tput_persisit.append(tput)
        self.episode.append(episode)
        self.time_persisit.append(time_duration)

    def learn(self,epoch_i,entropy_weight, IfPrint=True):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        _, loss,all_act_prob,entropy = self.sess.run([self.train_op,self.loss,self.all_act_prob,self.entropy],feed_dict={
            self.tf_obs:np.vstack(self.ep_obs),
            self.tf_acts:np.array(self.ep_as),
            self.tf_vt:discounted_ep_rs_norm,
            self.entropy_weight:entropy_weight
        })
        if IfPrint:
            print("epoch: %d, tput: %f, entropy: %f, loss: %f" % (
                epoch_i, np.mean(self.tput_batch), np.mean(entropy), np.mean(loss)))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.tput_batch = []
        self.entropy_persist.append(entropy)

    def _discount_and_norm_rewards(self):
        """
        Normalize reward per batch
        :return:
        """
        discounted_ep_rs = np.array(self.ep_rs)
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs) !=0:
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_session(self,ckpt_path):
        self.saver.save(self.sess, ckpt_path)

    def restore_session(self,ckpt_path):
        self.saver.restore(self.sess, ckpt_path)
