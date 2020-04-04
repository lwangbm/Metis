"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.
Policy Gradient, Reinforcement Learning.
"""
import os
import sys
# sys.path.append("/home/ubuntu/lra_scheduling_simulation/Simulation_Python")
sys.path.append("/home/lpwangthu/cpo")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbed.util.commons import *
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)
desired_kl = 0.0001
# safety_constraint = 100#0.00001#-0.00001
# safety_requirement = 0.5 #0.01
class PolicyGradient:

    def get_fisher_product_op(self):
        directional_gradients = tf.reduce_sum(self.kl_flat_gradients_op*self.vec)
        return get_flat_gradients(directional_gradients, self.trainable_variables)

    def get_fisher_product(self, vec, damping = 1e-3):
        self.feed_dict[self.vec] = vec
        return self.sess.run(self.fisher_product_op, self.feed_dict) + damping*vec

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 suffix="",
                 safety_requirement=0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.suffix = suffix

        self.safety_requirement = safety_requirement

        """
        self.ep_obs, self.ep_as, self.ep_rs: observation, action, reward, recorded for each batch
        """
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []

        """
        self.tput_batch: record throughput for each batch, used to show progress while training
        self.tput_persisit, self.episode: persist to record throughput, used to be stored and plot later

        """
        self.tput_batch, self.tput_persisit, self.safe_batch, self.safe_persisit = [], [], [], []
        self.coex_persisit, self.sum_persisit = [], []
        self.time_ersist = []
        self.node_used_persisit = []
        self.episode = []
        self.node_used = []
        self.start_cpo = False
        self.count = 0
        # TODO self.vio = []: violation

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        restore = ['Actor' + self.suffix + '/fc1' + self.suffix + '/kernel:0',
                   'Actor' + self.suffix + '/fc1' + self.suffix + '/bias:0',
                   'Actor' + self.suffix + '/fc2' + self.suffix + '/kernel:0',
                   'Actor' + self.suffix + '/fc2' + self.suffix + '/bias:0']
        restore_var = [v for v in tf.all_variables() if v.name in restore]
        self.saver = tf.train.Saver(var_list=restore_var)
        # self.saver = tf.train.Saver()

    def _build_net(self):

        with tf.variable_scope("Actor"+self.suffix):

            with tf.name_scope('inputs'+self.suffix):
                self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observation'+self.suffix)
                self.tf_acts = tf.placeholder(tf.int32, [None, ], name='actions_num'+self.suffix)
                self.tf_vt = tf.placeholder(tf.float32, [None, ], name='actions_value'+self.suffix)
                self.tf_safe = tf.placeholder(tf.float32, [None, ], name='safety_value'+self.suffix)
                self.entropy_weight = tf.placeholder(tf.float32, shape=(), name='entropy_weight_clustering'+self.suffix)

            # layer1 = tf.layers.dense(
            #      inputs=self.tf_obs,
            #      units=128,
            #      activation= tf.nn.tanh,
            #      kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
            #      bias_initializer= tf.constant_initializer(0.1),
            #      name='fc0'+self.suffix
            #  )
            layer = tf.layers.dense(
                inputs=self.tf_obs,
                units=128,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc1'+self.suffix
            )
            # layer3 = tf.layers.dense(
            #     inputs=layer,
            #     units=128 * 1,
            #     activation=tf.nn.tanh,
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='fc3' + self.suffix
            # )
            all_act = tf.layers.dense(
                inputs=layer,
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc2'+self.suffix
            )

            all_act2 = tf.layers.dense(inputs=all_act,
                                       units=self.n_actions,
                                       activation=None,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                       bias_initializer=tf.constant_initializer(0.1), name='fc22' + self.suffix)

            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor'+self.suffix+'/fc22')
            self.trainable_variables_shapes = [var.get_shape().as_list() for var in self.trainable_variables]

            # sampling
            self.all_act_prob = tf.nn.softmax(all_act2, name='act_prob'+self.suffix)
            self.all_act_prob = tf.clip_by_value(self.all_act_prob, 1e-20, 1.0)

            with tf.name_scope('loss'+self.suffix):
                neg_log_prob = tf.reduce_sum(-tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * tf.one_hot(indices=self.tf_acts, depth=self.n_actions), axis=1)
                loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
                loss += self.entropy_weight * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))
                self.entro = self.entropy_weight * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))
                self.loss = loss

            with tf.name_scope('train'+self.suffix):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

            # safety loss
            """
            * -1?
            """
            self.chosen_action_log_probs = tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * tf.one_hot(indices=self.tf_acts, depth=self.n_actions), axis=1)
            self.old_chosen_action_log_probs = tf.stop_gradient(tf.placeholder(tf.float32, [None]))
            # self.each_safety_loss = tf.exp(self.chosen_action_log_probs - self.old_chosen_action_log_probs) * self.tf_safe
            self.each_safety_loss = (tf.exp(self.chosen_action_log_probs) - tf.exp(self.old_chosen_action_log_probs)) * self.tf_safe
            self.average_safety_loss = tf.reduce_mean(self.each_safety_loss)  #/ self.n_episodes tf.reduce_sum
            # self.average_safety_loss +=self.entro

            # KL D
            self.old_all_act_prob = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.n_actions]))

            def kl(x, y):
                EPS = 1e-10
                x = tf.where(tf.abs(x) < EPS, EPS * tf.ones_like(x), x)
                y = tf.where(tf.abs(y) < EPS, EPS * tf.ones_like(y), y)
                X = tf.distributions.Categorical(probs=x + EPS)
                Y = tf.distributions.Categorical(probs=y + EPS)
                return tf.distributions.kl_divergence(X, Y, allow_nan_stats=False)


            self.each_kl_divergence = kl(self.all_act_prob, self.old_all_act_prob)  # tf.reduce_sum(kl(self.all_act_prob, self.old_all_act_prob), axis=1)
            self.average_kl_divergence = tf.reduce_mean(self.each_kl_divergence)
            # self.kl_gradients = tf.gradients(self.average_kl_divergence, self.trainable_variables)  # useless

            self.desired_kl = desired_kl
            self.metrics = [self.loss, self.average_kl_divergence, self.average_safety_loss, self.entro]

            # FLat
            self.flat_params_op = get_flat_params(self.trainable_variables)
            """not use tensorflow default function, here we calculate the gradient by self:
            (1) loss: g
            (2) kl: directional_gradients (math, fisher)
            (3) safe: b 
            """
            self.loss_flat_gradients_op = get_flat_gradients(self.loss, self.trainable_variables)
            self.kl_flat_gradients_op = get_flat_gradients(self.average_kl_divergence, self.trainable_variables)
            self.constraint_flat_gradients_op = get_flat_gradients(self.average_safety_loss, self.trainable_variables)

            self.vec = tf.placeholder(tf.float32, [None])
            self.fisher_product_op = self.get_fisher_product_op()

            self.new_params = tf.placeholder(tf.float32, [None])
            self.params_assign_op = assign_network_params_op(self.new_params, self.trainable_variables, self.trainable_variables_shapes)



            # with tf.name_scope('train'):
            #     self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
                # decay_rate =0.99999 # 0.999
                # learning_rate = 1e-1
                # self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})  # (4,) ->(1,4)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, prob_weights

    def choose_action_determine(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})  # (4,) ->(1,4)
        action = np.argmax(prob_weights.ravel())
        return action, prob_weights

    def store_training_samples_per_episode(self, s, a, r, ss):
        self.ep_obs.extend(s)
        self.ep_as.extend(a)
        self.ep_rs.extend(r)
        self.ep_ss.extend(ss)

    def store_tput_per_episode(self, tput, episode, list_check, coe_this, sum_this, duration=0):
        self.tput_batch.append(tput)
        self.tput_persisit.append(tput)
        self.episode.append(episode)
        self.safe_batch.append(list_check)
        self.safe_persisit.append(list_check)
        self.coex_persisit.append(coe_this)
        self.sum_persisit.append(sum_this)
        self.time_ersist.append(duration)


    def learn_vio (self, epoch_i, entropy_weight, IfPrint=False):
        discounted_ep_rs_norm = -self._discount_and_norm_safety()

        for _ in range(1):
            _, loss, all_act_prob, entropy = self.sess.run([self.train_op, self.loss, self.all_act_prob, self.entro], feed_dict={
                self.tf_obs: np.vstack(self.ep_obs),
                self.tf_acts: np.array(self.ep_as),
                self.tf_vt: discounted_ep_rs_norm,
                self.entropy_weight: entropy_weight
        })
        if IfPrint:

            print("epoch: %d, tput: %f, self.ep_ss: %f, safe_mean: %f, entro: %f, loss: %f" % (
                epoch_i, np.mean(self.tput_batch), np.mean(self.ep_ss), np.mean(self.safe_batch), np.mean(entropy), np.mean(loss)))

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []
        self.tput_batch, self.safe_batch = [], []

        return 0

    def learn(self, epoch_i, entropy_weight, Ifprint=False):

        # if np.mean(self.safe_batch) < 3.0:
        #     self.count += 1
        # else:
        #     self.count = 0
        # if self.count > 20:
        #     self.start_cpo = True
        # if not self.start_cpo:
        #     return self.learn_vio(epoch_i, entropy_weight, Ifprint)

        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        self.feed_dict = {self.tf_obs: np.vstack(self.ep_obs), self.tf_acts: np.array(self.ep_as),
                          self.tf_vt: discounted_ep_rs_norm, self.entropy_weight: entropy_weight,
                          # self.tf_safe: np.array(self.ep_ss)
                          self.tf_safe: self._discount_and_norm_safety()}

        chosen_action_log_probs = self.sess.run(self.chosen_action_log_probs, self.feed_dict)  # used in safe_loss
        self.feed_dict[self.old_chosen_action_log_probs] = chosen_action_log_probs  # same value, but stop gradient

        g, b, old_all_act_prob, old_params, old_safety_loss = self.sess.run(
            [self.loss_flat_gradients_op,
             self.constraint_flat_gradients_op,
             self.all_act_prob,
             self.flat_params_op,
             self.average_safety_loss],
            self.feed_dict)

        # kl diveregnce
        self.feed_dict[self.old_all_act_prob] = old_all_act_prob

        # math
        v = do_conjugate_gradient(self.get_fisher_product, g)  # x = A-1g
        # H_b = doConjugateGradient(self.getFisherProduct, b)
        approx_g = self.get_fisher_product(v)  # g = Ax = AA-1g
        # b = self.getFisherProduct(H_b)
        safety_constraint = self.safety_requirement - np.mean(self.safe_batch)
        linear_constraint_threshold = np.maximum(0, safety_constraint) + old_safety_loss
        eps = 1e-8
        delta = 2 * self.desired_kl
        c = -safety_constraint
        q = np.dot(approx_g, v)

        if (np.dot(b, b) < eps):
            lam = np.sqrt(q / delta)
            nu = 0
            w = 0
            r, s, A, B = 0, 0, 0, 0
            optim_case = 4
        else:
            norm_b = np.sqrt(np.dot(b, b))
            unit_b = b / norm_b
            w = norm_b * do_conjugate_gradient(self.get_fisher_product, unit_b)
            r = np.dot(w, approx_g)
            s = np.dot(w, self.get_fisher_product(w))
            A = q - (r ** 2 / s)
            B = delta - (c ** 2 / s)
            if (c < 0 and B < 0):
                optim_case = 3
            elif (c < 0 and B > 0):
                optim_case = 2
            elif (c > 0 and B > 0):
                optim_case = 1
            else:
                optim_case = 0
                return self.learn_vio(epoch_i, entropy_weight, Ifprint)
            lam = np.sqrt(q / delta)
            nu = 0

            if (optim_case == 2 or optim_case == 1):
                lam_mid = r / c
                L_mid = - 0.5 * (q / lam_mid + lam_mid * delta)

                lam_a = np.sqrt(A / (B + eps))
                L_a = -np.sqrt(A * B) - r * c / (s + eps)

                lam_b = np.sqrt(q / delta)
                L_b = -np.sqrt(q * delta)

                if lam_mid > 0:
                    if c < 0:
                        if lam_a > lam_mid:
                            lam_a = lam_mid
                            L_a = L_mid
                        if lam_b < lam_mid:
                            lam_b = lam_mid
                            L_b = L_mid
                    else:
                        if lam_a < lam_mid:
                            lam_a = lam_mid
                            L_a = L_mid
                        if lam_b > lam_mid:
                            lam_b = lam_mid
                            L_b = L_mid

                    if L_a >= L_b:
                        lam = lam_a
                    else:
                        lam = lam_b

                else:
                    if c < 0:
                        lam = lam_b
                    else:
                        lam = lam_a

                nu = max(0, lam * c - r) / (s + eps)

        if optim_case > 0:
            full_step = (1. / (lam + eps)) * (v + nu * w)

        else:
            full_step = np.sqrt(delta / (s + eps)) * w
        # print("optim_case: %f" %(optim_case))

        if (optim_case == 0 or optim_case == 1):
            new_params, status, new_kl_divergence, new_safety_loss, new_loss, entro = do_line_search_CPO(self.get_metrics, old_params, full_step, self.desired_kl, linear_constraint_threshold, check_loss=False)
        else:
            new_params, status, new_kl_divergence, new_safety_loss, new_loss, entro = do_line_search_CPO(self.get_metrics, old_params, full_step, self.desired_kl, linear_constraint_threshold)

        print('Success: ', status, "optim_case:", optim_case)

        if (status == False):
            self.sess.run(self.params_assign_op, feed_dict={self.new_params: new_params})


        # _, loss, all_act_prob, entro = self.sess.run([self.train_op, self.loss, self.all_act_prob, self.entro], feed_dict=self.feed_dict)
        if Ifprint:
            print("epoch: %d, tput: %f, self.ep_ss: %f, safe_mean: %f, new_kl_divergence: %f, new_safety_loss: %f, new_loss: %f, entro: %f" % (
                epoch_i, np.mean(self.tput_batch), np.mean(self.ep_ss), np.mean(self.safe_batch), new_kl_divergence, new_safety_loss, new_loss, entro))

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []
        self.tput_batch, self.safe_batch = [], []

        return optim_case


    def get_metrics(self, new_params):
        self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return self.sess.run(self.metrics, self.feed_dict)

    def _discount_and_norm_rewards(self):
        """
        Normalize reward per batch
        :return:
        """
        discounted_ep_rs = np.array(self.ep_rs)
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs) != 0:
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_session(self, ckpt_path):
        self.saver.save(self.sess, ckpt_path)

    def restore_session(self, ckpt_path):
        # restore = ['Actor' + self.suffix + '/fc1' + self.suffix + '/kernel:0', 'Actor' + self.suffix + '/fc1' + self.suffix + '/bias:0', 'Actor' + self.suffix + '/fc0' + self.suffix + '/kernel:0',
        #            'Actor' + self.suffix + '/fc0' + self.suffix + '/bias:0']
        # restore_var = [v for v in tf.all_variables() if v.name in restore]
        # self.saver = tf.train.Saver(var_list=restore_var)
        self.saver.restore(self.sess, ckpt_path)
        # self.saver = tf.train.Saver()

    def _discount_and_norm_safety(self):
        """
                Normalize safety violation per batch
                :return:
                """
        discounted_ep_ss = np.array(self.ep_ss)
        discounted_ep_ss -= np.mean(discounted_ep_ss)
        if np.std(discounted_ep_ss) != 0:
            discounted_ep_ss /= np.std(discounted_ep_ss)
        return discounted_ep_ss
