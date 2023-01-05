#coding=UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
import shutil
import sys
sys.path.append("..")
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.yolo.general_controller_yolo_multitask_continuous import GeneralController
from src.yolo.general_child_cal_yolo_continuous_mv import GeneralChild

from src.yolo.config import cfg
from src.yolo.dataset import Dataset

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("output_dir", "./output", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCHW'")
DEFINE_string("weight_path", "./output_yolo_robot_data", "")


DEFINE_integer("batch_size", 16, "")

DEFINE_integer("num_epochs", 310, "")
DEFINE_integer("child_lr_dec_every", 100, "")
DEFINE_integer("child_num_layers", 72, "")
DEFINE_integer("child_num_cells", 5, "")
DEFINE_integer("child_filter_size", 5, "")
DEFINE_integer("child_out_filters", 36, "")
DEFINE_integer("child_out_filters_scale", 1, "")
DEFINE_integer("child_num_branches", 5, "")
DEFINE_integer("child_num_cell_layers", 5, "")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_block_size", 3, "")
DEFINE_integer("child_lr_T_0", 10, "for lr schedule")
DEFINE_integer("child_lr_T_mul", 2, "for lr schedule")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_lr", 0.01, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.9, "")
DEFINE_float("child_drop_path_keep_prob", 0.6, "minimum drop_path_keep_prob")
DEFINE_float("child_l2_reg", 0.00004, "")
DEFINE_float("child_lr_max", 0.05, "for lr schedule")
DEFINE_float("child_lr_min", 0.0005, "for lr schedule")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_boolean("child_use_aux_heads", True, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")
DEFINE_float("child_moving_average", 0.9997, "")
DEFINE_boolean("child_epoch_lrdec", True, "")
# DEFINE_float("child_epoch_lrdec_ratio", [0.5, 0.75], "")

DEFINE_float("controller_lr", 0.001, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", 1.5, "")
DEFINE_float("controller_op_tanh_reduce", 2.5, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", 0.0001, "")
DEFINE_integer("controller_num_aggregate", 20, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_integer("controller_train_every", 1,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_search_whole_channels", True, "")
DEFINE_boolean("controller_sync_replicas", True, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")
DEFINE_boolean("controller_use_critic", False, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

tf.reset_default_graph()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_weight(weight_path):
  reader = tf.train.NewCheckpointReader(tf.train.latest_checkpoint(weight_path))
  # reader = tf.train.NewCheckpointReader(tf.train.latest_checkpoint('./model_ckpt'))
  return reader

def pruning(reader):
  all_variables = reader.get_variable_to_shape_map()
  all_variables_shape = {}
  for key_weight in all_variables:
    if (key_weight.split('/')[-1] == 'weight'):
      w_weight = reader.get_tensor(key_weight)
      all_variables[key_weight] = w_weight
      all_variables_shape[key_weight] = np.array(w_weight.shape)
      if ('_' in key_weight.split('/')[-2]):
        if ('bbox' in key_weight.split('/')[-2].split('_')[1]):
          key_biases = key_weight.split('weight')[0] + 'bias'
          w_biases = reader.get_tensor(key_biases)
          all_variables[key_biases] = w_biases
          all_variables_shape[key_biases] = np.array(w_biases.shape)
    elif (key_weight.split('/')[-1] == 'gamma'):
      key_biases = key_weight.split('gamma')[0] + 'beta'
      w_weight = reader.get_tensor(key_weight)
      w_biases = reader.get_tensor(key_biases)
      all_variables[key_weight] = w_weight
      all_variables[key_biases] = w_biases
      all_variables_shape[key_weight] = np.array(w_weight.shape)
      all_variables_shape[key_biases] = np.array(w_biases.shape)
  return all_variables,  all_variables_shape


weight_reader = load_weight(FLAGS.weight_path)
all_variables, all_variables_shape = pruning(weight_reader)
child_name = 'child'
epoch_lrdec_ratio = [0.5, 0.75]
first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS  # 20
second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS  # 30

def get_ops():
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """

  ControllerClass = GeneralController
  ChildClass = GeneralChild

  child_model = ChildClass(
    name=child_name,
    use_aux_heads=FLAGS.child_use_aux_heads,
    cutout_size=FLAGS.child_cutout_size,
    whole_channels=FLAGS.controller_search_whole_channels,
    num_layers=FLAGS.child_num_layers,
    num_cells=FLAGS.child_num_cells,
    num_branches=FLAGS.child_num_branches,
    fixed_arc=FLAGS.child_fixed_arc,
    out_filters_scale=FLAGS.child_out_filters_scale,
    out_filters=FLAGS.child_out_filters,
    keep_prob=FLAGS.child_keep_prob,
    drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
    num_epochs=FLAGS.num_epochs,
    l2_reg=FLAGS.child_l2_reg,
    data_format=FLAGS.data_format,
    batch_size=FLAGS.batch_size,
    clip_mode="norm",
    grad_bound=FLAGS.child_grad_bound,
    lr_init=FLAGS.child_lr,
    lr_dec_every=FLAGS.child_lr_dec_every,
    lr_dec_rate=FLAGS.child_lr_dec_rate,
    lr_cosine=FLAGS.child_lr_cosine,
    lr_max=FLAGS.child_lr_max,
    lr_min=FLAGS.child_lr_min,
    lr_T_0=FLAGS.child_lr_T_0,
    lr_T_mul=FLAGS.child_lr_T_mul,
    optim_algo="sgd",
    sync_replicas=FLAGS.child_sync_replicas,
    num_aggregate=FLAGS.child_num_aggregate,
    num_replicas=FLAGS.child_num_replicas,
    moving_average=FLAGS.child_moving_average,
    epoch_lrdec=FLAGS.child_epoch_lrdec,
    epoch_lrdec_ratio=epoch_lrdec_ratio,
  )

  if FLAGS.child_fixed_arc is None:
    controller_model = ControllerClass(
      res_id=[2, 5, 7, 10, 12, 14, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 50],
      search_whole_channels=FLAGS.controller_search_whole_channels,
      num_cells=FLAGS.child_num_cells,
      num_layers=FLAGS.child_num_layers,
      num_branches=FLAGS.child_num_branches,
      out_filters=FLAGS.child_out_filters,
      lstm_size=64,
      lstm_num_layers=1,
      lstm_keep_prob=1.0,
      tanh_constant=FLAGS.controller_tanh_constant,
      op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
      temperature=FLAGS.controller_temperature,
      lr_init=FLAGS.controller_lr,
      lr_dec_start=0,
      lr_dec_every=1000000,
      l2_reg=FLAGS.controller_l2_reg,
      entropy_weight=FLAGS.controller_entropy_weight,
      bl_dec=FLAGS.controller_bl_dec,
      use_critic=FLAGS.controller_use_critic,
      optim_algo="adam",
      sync_replicas=FLAGS.controller_sync_replicas,
      num_aggregate=FLAGS.controller_num_aggregate,
      num_replicas=FLAGS.controller_num_replicas)


    child_model.connect_controller(controller_model)
    controller_model.build_trainer(child_model)

    controller_ops = {
      "train_step": controller_model.train_step,
      "loss": controller_model.loss,
      "train_op": controller_model.train_op,
      "lr": controller_model.lr,
      "grad_norm": controller_model.grad_norm,
      "test_loss": controller_model.test_loss,
      "optimizer": controller_model.optimizer,
      "baseline": controller_model.baseline,
      "entropy": controller_model.sample_entropy,
      "sample_arc": controller_model.sample_arc,
    }
  else:
    assert not FLAGS.controller_training, (
      "--child_fixed_arc is given, cannot train controller")
    child_model.connect_controller(None)
    controller_ops = None

  child_ops = {
    "global_step": child_model.global_step,
    "loss": child_model.loss,
    "lr": child_model.learn_rate,
    "sample_arc": child_model.sample_arc,
    "flops": child_model.flops,
    "optimizer": child_model.optimizer,
    "if_enas": child_model.if_enas,
    "train_op": child_model.train_op_with_all_variables,
    "train_frozen_op": child_model.train_op_with_frozen_variables,
  }


  ops = {
    "child": child_ops,
    "controller": controller_ops,
  }

  return ops, child_model, controller_model


def train():
  trainset = Dataset('train')
  testset = Dataset('test')
  g = tf.Graph()
  with g.as_default():
    with tf.variable_scope(child_name, reuse=False):
      for key_weight in all_variables.keys():
        if key_weight.split('/')[-1] == 'weight':
          with tf.variable_scope(key_weight.split('/weight')[0].split('child/')[1], reuse=False):
            tf.get_variable('weight', trainable=True, shape=all_variables_shape[key_weight], initializer=tf.constant_initializer(all_variables[key_weight]))
            tf.get_variable('ini_weight', trainable=False, shape=all_variables_shape[key_weight], initializer=tf.constant_initializer(all_variables[key_weight]))
        elif key_weight.split('/')[-1] == 'bias':
          with tf.variable_scope(key_weight.split('/bias')[0].split('child/')[1], reuse=False):
            tf.get_variable('bias', trainable=True, shape=all_variables_shape[key_weight], initializer=tf.constant_initializer(all_variables[key_weight]))
            tf.get_variable('ini_bias', trainable=False, shape=all_variables_shape[key_weight],
                            initializer=tf.constant_initializer(all_variables[key_weight]))
        elif key_weight.split('/')[-1] == 'gamma':
          key_biases = key_weight.split('gamma')[0] + 'beta'
          with tf.variable_scope(key_weight.split('/gamma')[0].split('child/')[1], reuse=False):
            tf.get_variable('gamma', trainable=True, shape=all_variables_shape[key_weight], initializer=tf.constant_initializer(all_variables[key_weight]))
            tf.get_variable('beta', trainable=True, shape=all_variables_shape[key_biases], initializer=tf.constant_initializer(all_variables[key_biases]))
            tf.get_variable('ini_gamma', trainable=False, shape=all_variables_shape[key_weight], initializer=tf.constant_initializer(all_variables[key_weight]))
            tf.get_variable('ini_beta', trainable=False, shape=all_variables_shape[key_biases], initializer=tf.constant_initializer(all_variables[key_biases]))
    ops, child_model, controller_model = get_ops()
    child_ops = ops["child"]
    controller_ops = ops["controller"]

    saver = tf.train.Saver(max_to_keep=2)
    # checkpoint_saver_hook = tf.train.CheckpointSaverHook(
    #   FLAGS.output_dir, save_steps=trainset.num_batchs * trainset.batch_size, saver=saver)
    #
    # hooks = [checkpoint_saver_hook]
    # if FLAGS.child_sync_replicas:
    #   sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
    #   # sync_sessreplicas_hook = child_ops["optimizer"].make_session_run_hook(True)
    #   hooks.append(sync_replicas_hook)
    # if FLAGS.controller_training and FLAGS.controller_sync_replicas:
    #   sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
    #   hooks.append(sync_replicas_hook)

    # arc = np.zeros(49)
    arc = np.array([0.0 for i in range(72)])
    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    # with tf.train.SingularMonitoredSession(
    #   config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
    init_op = tf.global_variables_initializer()
    # with tf.train.MonitoredTrainingSession(
    #   config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
    with tf.Session() as sess:
        start_time = time.time()
        sess.run(init_op)
        epochs = 0
        while True:

          if_enas = 0

          for child_epoch in range(1, 1 + first_stage_epochs + second_stage_epochs):

            # pbar = tqdm(trainset)
            for train_data in trainset:
              if_enas += 1
              if child_epoch <= first_stage_epochs:

                run_ops = [
                  child_ops["loss"],
                  child_ops["lr"],
                  child_ops["flops"],
                  child_ops["sample_arc"],
                  child_ops["if_enas"],
                  child_ops['train_frozen_op'],
                ]

                feed_dict = {
                  child_model.input_data:   train_data[0],  # image
                  child_model.label_sbbox:  train_data[1],  # labels for compute loss
                  child_model.label_mbbox:  train_data[2],
                  child_model.label_lbbox:  train_data[3],
                  child_model.true_sbboxes: train_data[4],
                  child_model.true_mbboxes: train_data[5],
                  child_model.true_lbboxes: train_data[6],
                  child_model.sample_arc: arc,
                  child_model.if_enas: if_enas,
                  child_model.trainable: True,}

                loss, lr, flops, sample_arc, if_enas, _ = sess.run(run_ops, feed_dict=feed_dict)

              else:
                run_ops = [
                  child_ops["loss"],
                  child_ops["lr"],
                  child_ops["flops"],
                  child_ops["sample_arc"],
                  child_ops["if_enas"],
                  child_ops['train_op'],
                ]

                feed_dict = {
                  child_model.input_data: train_data[0],  # image
                  child_model.label_sbbox: train_data[1],  # labels for compute loss
                  child_model.label_mbbox: train_data[2],
                  child_model.label_lbbox: train_data[3],
                  child_model.true_sbboxes: train_data[4],
                  child_model.true_mbboxes: train_data[5],
                  child_model.true_lbboxes: train_data[6],
                  child_model.sample_arc: arc,
                  child_model.if_enas: if_enas,
                  child_model.trainable: True, }

                loss, lr, flops, sample_arc, if_enas, _ = sess.run(run_ops, feed_dict=feed_dict)

            global_step = sess.run(child_ops["global_step"])
            curr_time = time.time()
            log_string = ""
            log_string += "epoch={:<6d}".format(child_epoch)
            log_string += "step={}".format(global_step)
            log_string += " loss={:<8.6f}".format(loss)
            log_string += " lr={:<8.4f}".format(lr)
            log_string += " flops={}".format(flops)
            log_string += " sameple_arc={}".format(sample_arc)
            log_string += " if_enas={}".format(if_enas)
            log_string += " mins={:<10.2f}".format(
              float(curr_time - start_time) / 60)
            print(log_string)
          epochs += 1
          saver.save(sess, FLAGS.output_dir)

          test_epoch_loss = []
          for test_data in testset:
            test_step_loss = sess.run(child_model.loss, feed_dict={
              child_model.input_data: test_data[0],
              child_model.label_sbbox: test_data[1],
              child_model.label_mbbox: test_data[2],
              child_model.label_lbbox: test_data[3],
              child_model.true_sbboxes: test_data[4],
              child_model.true_mbboxes: test_data[5],
              child_model.true_lbboxes: test_data[6],
              child_model.sample_arc: arc,
              child_model.if_enas: if_enas,
              child_model.trainable: False,})
          test_epoch_loss.append(test_step_loss)
          test_epoch_loss = np.mean(test_epoch_loss)
          print('test_epoch_loss: ', test_epoch_loss)
          arc = sess.run(controller_ops["sample_arc"])
          num_blocks = [1, 2, 8, 8, 4]
          id = 1
          num_id = 1
          for t in range(len(num_blocks)):
            change_ids = np.reshape(np.arange(np.shape(arc)[0] + 1)[id:id + 2 * num_blocks[t] + 1:2], -1)
            sort_change_num = np.unique(np.reshape(np.sort(np.reshape(arc[num_id: change_ids[-1] + 1: 2], -1)), -1))
            change_num = sort_change_num[-1] if sort_change_num[-1] != 1.0 or np.shape(sort_change_num)[0] == 1 else \
            sort_change_num[-2]
            id += 2 * num_blocks[t] + 1
            num_id += 2 * num_blocks[t] + 1
            for i in change_ids:
              arc[i] = change_num if arc[i] != 1.0 else arc[i]

          print("Training controller")
          run_ops = [
            controller_ops["loss"],
            controller_ops["entropy"],
            controller_ops["lr"],
            controller_ops["grad_norm"],
            controller_ops["test_loss"],
            controller_ops["baseline"],
            controller_ops["train_op"],
          ]
          loss, entropy, lr, gn, test_loss, bl, _ = sess.run(run_ops, feed_dict={controller_model.test_loss: test_epoch_loss, controller_model.flops: flops / 1000000000.0})  # 训练控制器?
          controller_step = sess.run(controller_ops["train_step"])

          curr_time = time.time()
          log_string = ""
          log_string += "ctrl_step={:<6d}".format(controller_step)
          log_string += " loss={:<7.3f}".format(loss)
          log_string += " ent={:<5.2f}".format(entropy)
          log_string += " lr={:<6.4f}".format(lr)
          log_string += " |g|={:<8.4f}".format(gn)
          log_string += " test_loss={:<6.4f}".format(test_loss)
          log_string += " bl={:<5.2f}".format(bl)
          log_string += " mins={:<.2f}".format(
              float(curr_time - start_time) / 60)
          print(log_string)

          if epochs >= FLAGS.num_epochs:
            break


def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  utils.print_user_flags()
  train()


if __name__ == "__main__":
  tf.app.run()

