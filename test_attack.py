## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from l0_attack import CarliniL0
from l0_attack_batch import CarliniL0_batch
from LPA_attack import LPA_attack
from LPA_attack2 import LPA_attack2
from LPA_attack_batch import LPA_attack_batch

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
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
    labels = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = np.random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
            labels.append(np.argmax(data.test_labels[start + i]))
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    labels = np.array(labels)
    return inputs, targets, labels

if __name__ == "__main__":
    dataset = "mnist"#"cifar"
    Targeted = True
    Iterations = 500
    with tf.Session() as sess:
        if dataset == "mnist":
            data = MNIST()
            model = MNISTModel("/models/mnist", sess)
        elif dataset == "cifar":
            data = CIFAR()
            model = CIFARModel("models/cifar",sess)
        else:
            raise Exception("Invalid dataset!", dataset)

        # attack = CarliniL0(sess, model,targeted = Targeted,max_iterations=1000)
        # attack = Leastpixel_attack(sess, model, max_iterations=1000)
        attack = CarliniL0_batch(sess, model,targeted = Targeted, max_iterations=1000)
        # attack = LPA_attack(sess, model, max_iterations=2000, targeted=Targeted)
        # attack = LPA_attack2(sess, model, max_iterations=4000,targeted=Targeted)
        # attack = LPA_attack_batch(sess, model, max_iterations=2000,targeted=Targeted)
        if Targeted:
            length = 10
        else:
            length = 1
        cat_count = np.zeros([length])
        label_count = np.zeros([length])
        perturbation = np.zeros([length])
        fake_count = np.zeros([length])
        timestart = time.time()
        for i in range(Iterations):

            inputs, targets ,label= generate_data(data, samples=1, targeted=Targeted,start=i, inception=False)
            print(np.array(inputs).shape)
            for j in range(len(label)):
              label_count[label[j]] = label_count[label[j]] + 1
            adv,cat_count,fake_count,perturbation = attack.attack(inputs, targets,cat_count,perturbation,fake_count)
            print("label",i, label_count)
            print("cat",i,cat_count)
            print("fake",i,fake_count)
            print("perturbation",i,perturbation)
        timeend = time.time()
        print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")





