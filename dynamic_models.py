import numpy as np
import gin
from multiprocessing import Pool
from physbam_python.rollout_physbam import rollout_single as rollout_single_2d
from physbam_python_new.rollout_physbam import rollout_single as rollout_single_3d


def _pickle_method(method):
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
		cls_name = cls.__name__.lstrip('_')
		func_name = '_' + cls_name + func_name
	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	for cls in cls.__mro__:
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	return func.__get__(obj, cls)

import copyreg
import types
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


@gin.configurable
class physbam_2d(object):
  def __init__(self, physbam_args=" -disable_collisions -stiffen_bending 100"):
    self.physbam_args = physbam_args
#    self.pool = Pool(8)

  def execute(self, state, actions):
    """Execute action sequence and get end state.
    Args:
      state: (N,2) array representing the current state.
      actions: A list of (Int, array) tuple, where the int is
          the action node, and the array is 2d vector of action.
    """
    for ac in actions:
      state = rollout_single_2d(state, ac[0]+1, ac[1], frames=1,
                               physbam_args=self.physbam_args)
    return state

  def execute_batch(self, state, actions):
    """Execute in batch.
    Args:
      state: (N,2) array or a list of (N,2) array representing the current state.
      actions: A list of list of (Int, array) tuple.
    """
    if isinstance(state, list):
        assert(len(state)==len(actions))
    else:
        assert(state.ndim==2)
        state = [state for a in actions]
    pool = Pool(8)
    states = pool.starmap(self.execute, zip(state, actions))
    pool.close()
    pool.join()
    return states

  def __getstate__(self):
    self_dict = self.__dict__.copy()
#    del self_dict['pool']
    return self_dict

  def __setstate__(self, state):
    self.__dict__.update(state)


@gin.configurable
class physbam_3d(object):
  def __init__(self, physbam_args=" -friction 0.176 -stiffen_linear 2.223 -stiffen_bending 0.218"):
    self.physbam_args = physbam_args

  def execute(self, state, actions):
    """Execute action sequence and get end state.
    Args:
      current: (N,2) array representing the current state.
      actions: A list of (Int, array) tuple, where the int is
          the action node, and the array is 2d vector of action.
    """
    # transform actions
    actions = [[ac[1][0], ac[1][1], float(ac[0])/(state.shape[0]-1)] for ac in actions]
    actions = np.array(actions)
    state = rollout_single_3d(state, actions, physbam_args=' -dt 1e-3 ' + self.physbam_args)
    return state

  def execute_batch(self, state, actions):
    """Execute in batch.
    Args:
      state: (N,2) array or a list of (N,2) array representing the current state.
      actions: A list of list of (Int, array) tuple.
    """
    if isinstance(state, list):
        assert(len(state)==len(actions))
    else:
        assert(state.ndim==2)
        state = [state for a in actions]
    pool = Pool(8)
    states = pool.starmap(self.execute, zip(state, actions))
    pool.close()
    pool.join()
    return states


import tensorflow as tf
from neural_simulator.model_wrapper import Model


@gin.configurable
class neural_sim(object):
  def __init__(self, model_type, snapshot):
    self.start = tf.placeholder(tf.float32, shape=[None, 64, 2])
    self.action = tf.placeholder(tf.float32, shape=[None, 64, 2])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    self.sess = tf.Session(config=tf_config)
    self.model = Model(model_type)
    self.model.build(input=self.start, action=self.action)
    self.sess.run(tf.global_variables_initializer())
    self.model.load(self.sess, snapshot)
#    print("Warning: This model is hacked to work with state and action in robot coordinate.")
#    print("It will convert input state and action to fit the distribution of training data")

  def execute(self, state, actions):
#    state = state*np.array([-1.0, 1.0])+np.array([0.5, 0.0])
#    actions = [(a[0], np.array([-a[1][0], a[1][1]])) for a in actions]

    onehot_actions = []
    for ac in actions:
      onehot_action = np.zeros_like(state)
      onehot_action[ac[0],:] = ac[1]
      onehot_actions.append(onehot_action)
    for ac in onehot_actions:
      state = self.model.predict_single(self.sess, state, ac)
#    state[:,0]=0.5-state[:,0]
    return state

  def execute_batch(self, state, actions):
    if isinstance(state, list) or state.ndim==3:
        assert(len(state)==len(actions))
    else:
        assert(state.ndim==2)
        state = [state for a in actions]
    state = np.array(state)

#    state = state*np.array([-1.0, 1.0])+np.array([0.5, 0.0])
#    actions = [[(a[0], np.array([-a[1][0], a[1][1]])) for a in action] for action in actions]

    onehot_actions = []
    num_steps = len(actions[0]) # assume it's the same for each
    for t in range(num_steps):
      onehot_ac = np.zeros_like(state)
      for i,action in enumerate(actions):
        onehot_ac[i,action[t][0],:] = action[t][1]
      onehot_actions.append(onehot_ac)
    for ac in onehot_actions:
      state = self.model.predict_batch(self.sess, state, ac)
#    state[:,:,0]=0.5-state[:,:,0]
    return state


if __name__=='__main__':
    sc = physbam_2d()
    state = np.zeros((64,2))
    actions = [[( 0,np.zeros((2,)) )]]
    sc.execute_batch(state, actions)
