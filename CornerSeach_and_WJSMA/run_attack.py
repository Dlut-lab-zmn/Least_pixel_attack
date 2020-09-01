import tensorflow as tf
import scipy.io
import numpy as np
import argparse
from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from numpy import random
import time
def load_model_and_dataset(dataset):
  if dataset == 'mnist':
    import mnist_NiN_bn
    model = mnist_NiN_bn.NiN_Model()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint('/home/bull/home/zmn/insight/sparse-imperceivable-attacks-master/models/mnist_NiN/')
    saver.restore(sess, checkpoint)
    data = MNIST()
  elif dataset == "mnist2":
    import mnist_model
    model = mnist_model.MNISTModel()
    data = MNIST()
  elif dataset == 'cifar10':
    import cifar_NiN_bn
    model = cifar_NiN_bn.NiN_Model()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint('/home/bull/home/zmn/insight/sparse-imperceivable-attacks-master/models/cifar_NiN/')
    saver.restore(sess, checkpoint)

    data = CIFAR()
  else:
    raise ValueError('unknown dataset')
    
  return model,data
def generate_data(data, samples, targeted=False, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    #labels = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])
            inputs.append(data.test_data[start + i])
            targets.append(9)
            #labels.append(np.argmax(data.test_labels[start + i]))
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])
            targets = np.argmax(np.array(targets),1)
    inputs = np.array(inputs)
    
    #labels = np.array(labels)
    return inputs, targets#, labels

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Define hyperparameters.')
  parser.add_argument('--dataset', type=str, default='mnist2', help='cifar10, mnist')
  parser.add_argument('--attack', type=str, default='CS', help='PGD, CS')
  parser.add_argument('--path_results', type=str, default='none')

  hps = parser.parse_args()
  
  sess = tf.InteractiveSession()
  model,data = load_model_and_dataset(hps.dataset)

  if hps.attack == 'PGD':
    import pgd_attacks
    
    args = {'type_attack': 'L0+sigma',
            'n_restarts': 3,
            'num_steps': 20,
            'step_size': 120000.0/255.0/2.0,
            'kappa': 0.8,
            'epsilon': -1,
            'sparsity': 50}
            
    attack = pgd_attacks.PGDattack(model, args)
  
  elif hps.attack == 'CS':
    import cornersearch_attacks
    
    args = {'type_attack': 'L0',
            'n_iter': 1000,
            'n_max': 150,
            'kappa': 0.8,
            'epsilon': -1,
            'sparsity': 100,
            'size_incr': 5}#sigma
    
    attack = cornersearch_attacks.CSattack(model, args)
  overall_points = 0
  fake = 0
  iters = 500
  Targeted_attack = True
  s_time = time.time()
  for i in range(iters):
    input, label = generate_data(data, samples=1, targeted=Targeted_attack,start=i, inception=False)
    corr_pred = sess.run(model.predictions, {model.x_input: input, model.y_input: label})
    # x_test, y_test are images and labels on which the attack is run (to be loaded)
    # x_test in the format bs (batch size) x heigth x width x channels
    # y_test in the format bs
    if hps.attack == 'PGD':
        adv, pgd_adv_acc = attack.perturb(input, label, sess)
    elif hps.attack == 'CS':
        adv, pixels_changed, fl_success,fake,overall_points = attack.perturb(input, label,fake,overall_points, sess,targeted = Targeted_attack)
    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)
  print(overall_points)
  print(fake)
  print(time.time()-s_time)
  sess.close()