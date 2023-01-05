#coding=UTF-8
import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.controller import Controller
from src.yolo.utils import get_train_ops
from src.common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class GeneralController(Controller):
  def __init__(self,
               res_id,
               search_for="macro",
               search_whole_channels=False,
               num_layers=4,
               num_branches=6,
               out_filters=48,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               tanh_constant=None,
               temperature=None,
               lr_init=1e-3,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               l2_reg=0,
               entropy_weight=None,
               clip_mode=None,
               grad_bound=None,
               use_critic=False,
               bl_dec=0.999,
               optim_algo="adam",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               name="controller",
               *args,
               **kwargs):

    print("-" * 80)
    print("Building ConvController")

    self.search_for = search_for
    self.search_whole_channels = search_whole_channels
    self.num_layers = num_layers
    self.num_branches = num_branches
    self.out_filters = out_filters

    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers 
    self.lstm_keep_prob = lstm_keep_prob
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.l2_reg = l2_reg
    self.entropy_weight = entropy_weight
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.use_critic = use_critic
    self.bl_dec = bl_dec


    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.name = name

    self.res_id = res_id

    self._create_params()
    self._build_sampler()


  def _create_params(self):
    initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    with tf.variable_scope(self.name, initializer=initializer):
      with tf.variable_scope("lstm"):
        self.w_lstm = []
        for layer_id in range(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w = tf.get_variable(
              "w", [2 * self.lstm_size, 4 * self.lstm_size])
            self.w_lstm.append(w)

      self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])
      if self.search_whole_channels:
        with tf.variable_scope("emb"):
          self.w_emb = tf.get_variable(
            "w", [self.num_branches, self.lstm_size])
          self.w_resemb = tf.get_variable(
            "w_res", [2, self.lstm_size])
        with tf.variable_scope("softmax"):
          self.w_soft = tf.get_variable(
            "w", [self.lstm_size, self.num_branches])
          self.w_ressoft = tf.get_variable(
            "w_res", [self.lstm_size, 2])

      else:
        self.w_emb = {"start": [], "count": []}
        with tf.variable_scope("emb"):
          for branch_id in range(self.num_branches):
            with tf.variable_scope("branch_{}".format(branch_id)):
              self.w_emb["start"].append(tf.get_variable(
                "w_start", [self.out_filters, self.lstm_size]));
              self.w_emb["count"].append(tf.get_variable(
                "w_count", [self.out_filters - 1, self.lstm_size]));

        self.w_soft = {"start": [], "count": []}
        with tf.variable_scope("softmax"):
          for branch_id in range(self.num_branches):
            with tf.variable_scope("branch_{}".format(branch_id)):
              self.w_soft["start"].append(tf.get_variable(
                "w_start", [self.lstm_size, self.out_filters]));
              self.w_soft["count"].append(tf.get_variable(
                "w_count", [self.lstm_size, self.out_filters - 1]));

      with tf.variable_scope("attention"):
        self.w_attn_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
        self.w_attn_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
        self.v_attn = tf.get_variable("v", [self.lstm_size, 1])

  def _sample_branch_id(self, inputs, prev_c, prev_h):
    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
    prev_c, prev_h = next_c, next_h
    logit = tf.matmul(next_h[-1],
                      self.w_soft)
    if self.temperature is not None:
      logit /= self.temperature
    if self.tanh_constant is not None:
      logit = self.tanh_constant * tf.tanh(logit)
    branch_id = tf.multinomial(logit, 1)
    branch_id = tf.to_int32(branch_id)
    branch_id = tf.reshape(branch_id, [1])
    log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=branch_id)
    entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
    inputs = tf.nn.embedding_lookup(self.w_emb, branch_id)
    return branch_id, prev_c, prev_h, inputs, log_prob, entropy

  def _inherite_tensors(self, branch_id, prev_c, prev_h, inputs, log_prob, entropy):
    return branch_id, prev_c, prev_h, inputs, log_prob, entropy

  def _sample_if_resprune(self, inputs, prev_c, prev_h):
    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
    prev_c, prev_h = next_c, next_h
    logit = tf.matmul(next_h[-1], self.w_ressoft)
    if self.temperature is not None:
      logit /= self.temperature
    if self.tanh_constant is not None:
      logit = self.tanh_constant * tf.tanh(logit)
    if_prune = tf.multinomial(logit, 1)
    if_prune = tf.to_int32(if_prune)
    if_prune = tf.reshape(if_prune, [1])
    res_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=if_prune)
    res_entropy = tf.stop_gradient(res_log_prob * tf.exp(-res_log_prob))
    inputs = tf.nn.embedding_lookup(self.w_resemb, if_prune)
    branch_id = tf.cond(tf.equal(if_prune[0], 1), lambda: tf.Variable([5], trainable=False), lambda: tf.Variable([1], trainable=False))
    log_prob = tf.cond(tf.equal(if_prune[0], 1), lambda: tf.Variable([0.0], trainable=False), lambda: tf.Variable([1.0], trainable=False))
    entropy = tf.cond(tf.equal(if_prune[0], 1), lambda: tf.Variable([0.0], trainable=False), lambda: tf.Variable([1.0], trainable=False))

    branch_id, prev_c, prev_h, inputs, log_prob, entropy = tf.cond(tf.equal(if_prune[0], 1), lambda: self._inherite_tensors(branch_id, prev_c, prev_h, inputs, log_prob, entropy), lambda: self._sample_branch_id(inputs, prev_c, prev_h))
    return branch_id, prev_c, prev_h, inputs, log_prob, entropy, res_log_prob, res_entropy, if_prune

  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""

    print("-" * 80)
    print("Build controller sampler")


    arc_seq = []
    entropys = []
    log_probs = []
    res_prune = []
    res_log_probs = []
    res_entropys = []
    # skip_count = []
    # skip_penaltys = []

    prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
              range(self.lstm_num_layers)]
    prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
              range(self.lstm_num_layers)]
    inputs = self.g_emb
    for layer_id in range(self.num_layers):
      if layer_id in self.res_id:
        print(1)
        branch_id, prev_c, prev_h, inputs, log_prob, entropy, res_log_prob, res_entropy, if_prune = self._sample_if_resprune(inputs, prev_c, prev_h)
        res_prune.append(if_prune)
        arc_seq.append(branch_id)
        log_probs.append(log_prob)
        entropys.append(entropy)
        res_log_probs.append(res_log_prob)
        res_entropys.append(res_entropy)


      elif layer_id - 1 in self.res_id:
        print(2)
        branch_id, prev_c, prev_h, inputs, log_prob, entropy = tf.cond(tf.equal(if_prune[0], 1),
                                                                     lambda: self._inherite_tensors(branch_id, prev_c,
                                                                                                    prev_h, inputs,
                                                                                                    log_prob, entropy),
                                                                     lambda: self._sample_branch_id(inputs, prev_c, prev_h))
        arc_seq.append(branch_id)
        log_probs.append(log_prob)
        entropys.append(entropy)


      else:
        print(3)
        branch_id, prev_c, prev_h, inputs, log_prob, entropy = self._sample_branch_id(inputs, prev_c, prev_h)
        arc_seq.append(branch_id)

    arc_seq = tf.concat(arc_seq, axis=0)
    self.sample_arc = tf.reshape(arc_seq, [-1])

    entropys = tf.stack(entropys)
    self.sample_entropy = tf.reduce_sum(entropys)
    res_entropys = tf.stack(res_entropys)
    self.sample_entropy = tf.add(self.sample_entropy, tf.reduce_sum(res_entropys))

    log_probs = tf.stack(log_probs)
    self.sample_log_prob = tf.reduce_sum(log_probs)
    res_log_probs = tf.stack(res_log_probs )
    self.sample_log_prob = tf.add(self.sample_log_prob, tf.reduce_sum(res_log_probs))


  def build_trainer(self, child_model):
    self.test_loss = tf.placeholder(dtype=tf.float32, shape=(), name="reward_acc")
    self.flops = tf.placeholder(dtype=tf.float32, shape=(), name="flops")

    self.reward = -(self.test_loss + self.flops)

    if self.entropy_weight is not None:
      self.reward += self.entropy_weight * self.sample_entropy
    self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    baseline_update = tf.assign_sub(
      self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))

    with tf.control_dependencies([baseline_update]):
      self.reward = tf.identity(self.reward)
    self.reward = tf.identity(self.reward)
    self.loss = self.sample_log_prob * (self.reward - self.baseline)


    self.train_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="train_step")
    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    print("-" * 80)
    for var in tf_variables:
      print(var)

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.train_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas,
      if_child=False)

