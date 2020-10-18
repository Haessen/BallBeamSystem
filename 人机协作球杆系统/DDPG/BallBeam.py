import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
np.random.seed(1)
tf.set_random_seed(1)

LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.1),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]  # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a_, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)  # 权重参数矩阵W初始化
            init_b = tf.constant_initializer(0.1)  # 矩阵b初始化
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]  # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('l2'):
                n_l2 = 40
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=init_w, trainable=trainable)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
                net2 = tf.nn.relu(tf.matmul(net, w2) + b2)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net2, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)
        return q  # Q(s,a)

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

    def get_qvalue(self, s, a):
        return sess.run(self.q, feed_dict={S: s, self.a: a})


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


env = gym.make('BallBeam-v0')
state_dim = 4
action_dim = 1
max_action = 10.0 / 180 * np.pi  # 单位为弧度

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard:
actor = Actor(sess, action_dim, max_action, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

saver = tf.train.Saver()  # 模型持久化

sess.run(tf.global_variables_initializer())
M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

MAX_EPISODES = 200
MAX_EP_STEPS = 200
var = 0.9
render_FLAG = False

count_Flag = False
learn_epi_count = 0
# 画图曲线
epi_rewards = np.zeros(MAX_EPISODES)
epi_step = np.zeros(MAX_EPISODES)

test_state1 = np.array([0, 0.5, 0, 0])
test_state2 = np.array([0, 0.5, 0.1, 0])
test_state3 = np.array([0, 0.5, -0.1, 0])
test_state4 = np.array([0, 0.5, 0, 5 / 180 * np.pi])
test_state5 = np.array([0, 0.5, 0, -5 / 180 * np.pi])
test_action = np.zeros((MAX_EPISODES, 5))
test_action_value = np.zeros((MAX_EPISODES, 5))
test_take_action = np.array([0, -5 / 180 * 3.14, -10 / 180 * 3.14, 5 / 180 * 3.14, 10 / 180 * 3.14]).reshape(5,
                                                                                                             action_dim)

for i in range(MAX_EPISODES):
    print('in epi ', i)
    s = env.reset()
    ep_reward = 0
    # env.render()
    j = 0
    test_action[i, 0] = actor.choose_action(test_state1)
    test_action[i, 1] = actor.choose_action(test_state2)
    test_action[i, 2] = actor.choose_action(test_state3)
    test_action[i, 3] = actor.choose_action(test_state4)
    test_action[i, 4] = actor.choose_action(test_state5)
    test_action_value[i, 0] = critic.get_qvalue(test_state1.reshape((1, state_dim)),
                                                test_take_action[0, 0].reshape(1, action_dim))
    test_action_value[i, 1] = critic.get_qvalue(test_state1.reshape((1, state_dim)),
                                                test_take_action[1, 0].reshape(1, action_dim))
    test_action_value[i, 2] = critic.get_qvalue(test_state1.reshape((1, state_dim)),
                                                test_take_action[2, 0].reshape(1, action_dim))
    test_action_value[i, 3] = critic.get_qvalue(test_state1.reshape((1, state_dim)),
                                                test_take_action[3, 0].reshape(1, action_dim))
    test_action_value[i, 4] = critic.get_qvalue(test_state1.reshape((1, state_dim)),
                                                test_take_action[4, 0].reshape(1, action_dim))
    for j in range(MAX_EP_STEPS):
        if render_FLAG is True:
            pic_data = env.render('rgb_array')

        print('in step ', j)
        a = actor.choose_action(s)
        if (np.random.uniform()) < var:
            a = np.random.uniform(-max_action, max_action, (1,))
        a = np.clip(a, -max_action, max_action)
        s_, r, done, info = env.step(a)
        print('s ', s)
        print('a ', a)
        print('r ', r)
        print('var is ', var)
        M.store_transition(s, a, r, s_)
        if M.pointer > MEMORY_CAPACITY:
            print('learning')
            if count_Flag is False:
                learn_epi_count = i
                count_Flag = True
            render_FLAG = True
            if var > 0.1:
                var *= .9995  # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)
        s = s_
        ep_reward += r
        if done is True:
            print('Be Terminated')
            epi_rewards[i] = ep_reward / j
            epi_step[i] = j
            break
        epi_rewards[i] = ep_reward / j
        epi_step[i] = j

print('learn epi is ', learn_epi_count)

# 绘图
plt.subplot(121)
plt.plot(np.arange(MAX_EPISODES), epi_rewards)
plt.xlabel('episodes\n(a)')
plt.ylabel('rewards')

plt.subplot(122)
plt.plot(np.arange(MAX_EPISODES), epi_step)
plt.xlabel('episodes\n(b)')
plt.ylabel('step')
plt.show()
