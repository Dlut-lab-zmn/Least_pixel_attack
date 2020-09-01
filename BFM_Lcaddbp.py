## l0_attack.py -- attack a network optimizing for l_0 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
from numpy import random

BATCH_SIZE = 30
MAX_ITERATIONS = 1000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # abort gradient descent upon first valid solution
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results


class BFM_Lcaddbp:
    def __init__(self, sess, model,
                 learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 targeted=True,
                 independent_channels=False):
        """
        The L_0 optimized attack.

        Returns adversarial examples for the supplied model.

        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        independent_channels: set to false optimizes for number of pixels changed,
          set to true (not recommended) returns number of channels changed.
        """

        self.model = model
        self.sess = sess

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.independent_channels = independent_channels

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

        self.grad = self.gradient_descent(sess, model)

    def L_0loss(self, x, beta=10, alpha=1):
        return alpha * tf.reduce_sum(alpha - alpha / tf.exp(tf.square(x) * beta), (1, 2, 3))

    def norm_to_01(self, x, beta=1, alpha=1):
        return x#1-  1/tf.exp(0.01*x)#1- alpha / tf.exp(tf.square(x) * beta)

    def gradient_descent(self, sess, model):
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y

        shape = (BATCH_SIZE, model.image_size, model.image_size, model.num_channels)

        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))
        canchange = tf.Variable(np.zeros(shape), dtype=np.float32)
        # the variables we're going to hold, use for efficiency

        simg = tf.Variable(np.zeros(shape, dtype=np.float32))
        original = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape, dtype=np.float32))
        tlab = tf.Variable(np.zeros((BATCH_SIZE, model.num_labels), dtype=np.float32))

        # and the assignment to set the variables
        assign_simg = tf.placeholder(np.float32, shape)
        assign_original = tf.placeholder(np.float32, shape)
        assign_timg = tf.placeholder(np.float32, shape)
        assign_tlab = tf.placeholder(np.float32, (BATCH_SIZE, self.model.num_labels))

        # these are the variables to initialize when we run
        setup = []
        setup.append(tf.assign(timg, assign_timg))
        setup.append(tf.assign(original, assign_original))
        setup.append(tf.assign(simg, assign_simg))
        setup.append(tf.assign(tlab, assign_tlab))

        newimg = (tf.tanh(modifier + simg) / 2) * self.norm_to_01(canchange) + (1 - self.norm_to_01(canchange)) * original
        Initnewimg = newimg
        Initnewimg = tf.clip_by_value((Initnewimg + 0.5) * 255., 0., 255.)
        Initnewimg = Initnewimg / 255. - 0.5


        Initoutput = model.predict(Initnewimg)

        Initreal = tf.reduce_sum((tlab) * Initoutput, 1)

        Initother = tf.reduce_max((1 - tlab) * Initoutput - (tlab * 10000), 1)

        if self.TARGETED:
            Initloss1 = tf.maximum(0.0, Initother - Initreal + .01)
        else:
            Initloss1 = tf.maximum(0.0, Initreal - Initother + .01)

        # sum up the losses
        Initloss_sbin = self.L_0loss(self.norm_to_01(canchange),10.)
        
        Initloss_midbin = tf.where(tf.is_nan(Initloss_sbin), tf.zeros_like(Initloss_sbin), Initloss_sbin)
        Initloss_sbin = tf.where(tf.is_nan(Initloss_sbin), tf.zeros_like(Initloss_sbin)+tf.reduce_mean(Initloss_midbin), Initloss_sbin)
        
        Initloss_bin =Initloss_sbin# tf.reduce_mean(Initloss_sbin)#

        Initloss_smod = self.L_0loss((tf.tanh(modifier + simg) / 2 - tf.tanh(timg) / 2),10.)
        Initloss_midbin = tf.where(tf.is_nan(Initloss_smod), tf.zeros_like(Initloss_smod), Initloss_smod)
        Initloss_smod = tf.where(tf.is_nan(Initloss_smod), tf.zeros_like(Initloss_smod)+tf.reduce_mean(Initloss_midbin), Initloss_smod)
        Initloss_mod = Initloss_smod

        Initloss =10.*Initloss1 + 0.5* Initloss_bin+ 0.5* Initloss_mod# 0.2 for mnist, 0.5 for cifar10

        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        Inittrain = optimizer.minimize(Initloss, var_list=[modifier, canchange])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier, canchange, simg,
                                                  original, timg, tlab] + new_vars)

        def doit(oimgs, labs, starts):
            # convert to tanh-space
            imgs = np.arctanh(np.array(oimgs) * 1.999999)
            starts = np.arctanh(np.array(starts) * 1.999999)
            # initialize the variables
            sess.run(init)
            sess.run(setup, {assign_timg: imgs,
                             assign_tlab: labs,
                             assign_simg: starts,
                             assign_original: oimgs})


            old_nimg = None
            old_Equal_count = old_Initloss_b = 1000.
            for step in range(self.MAX_ITERATIONS):
                    # remember the old value

                    _, works, ploss, qloss, Initloss_b, Initloss_m = sess.run(
                        [Inittrain, Initloss1, Initoutput, tlab, Initloss_sbin, Initloss_mod])
                    #print("works",works)
                    #print("Initloss_b",Initloss_b)
                    #print("Initloss_m",Initloss_m)
                    if self.TARGETED:
                        Flag = np.argmax(ploss,1) == np.argmax(np.squeeze(qloss))
                        
                        if np.sum(Flag) >= 1:
                            op_index = np.argmin(Initloss_b * Flag)
                            nimg = sess.run((Initnewimg))
                            if Initloss_b[op_index] < old_Initloss_b:
                                old_nimg = nimg[op_index]
                                old_Initloss_b = Initloss_b[op_index]
                    else:
                        if np.argmax(ploss) != np.argmax(np.squeeze(qloss)):
                            nimg = sess.run((Initnewimg))
                            cal_img = np.around(np.clip((np.array(oimgs) + 0.5) * 255., 0., 255.))
                            cal_nimg = np.around(np.clip((np.array(nimg) + 0.5) * 255., 0., 255.))
                            Equal_count = np.sum(np.all(np.abs(cal_img - cal_nimg) > 1, axis=3), (1, 2))
                            if Equal_count < old_Equal_count:
                                old_Equal_count = Equal_count
                                old_nimg = nimg
            if old_nimg is not None:
                init_input=  np.expand_dims(oimgs[0],0)
                cal_img = np.around(np.clip((init_input + 0.5) * 255., 0., 255.))
                cal_nimg = np.around(np.clip((np.array(old_nimg) + 0.5) * 255., 0., 255.))
                Equal_count = np.sum(np.abs(cal_img - cal_nimg) > 1.)

                #print("Equal count:", np.sum(np.all(np.sum(np.abs(cal_img - cal_nimg), 0) > 1, axis=2)))
                return Equal_count, old_nimg
            else:
                return None,None

        return doit

    def attack(self, imgs, targets, cat_count,perturbation,fake_count):
        """
        Perform the L_0 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        for i, (img, target) in enumerate(zip(imgs, targets)):
            #print("Attack iteration", i)
            img_set = []
            tar_set = []
            nimg_set = []
            noise2 = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9]) - 0.5
            for b in range(BATCH_SIZE):
                noise = (random.rand(self.model.image_size, self.model.image_size,
                                     self.model.num_channels) - 0.5)  # *10./ (255.)
                # nimg_set.append(np.clip( img + noise2[b] / 255., -0.5, 0.5))
                nimg_set.append(np.clip(img+noise, -0.5, 0.5))  # img+noise
                img_set.append(img)
                tar_set.append(target)
            equal_count,newimg = self.attack_single(nimg_set, img_set, tar_set)
            if equal_count is not None:
                #print(equal_count)
                cat_count[i] = cat_count[i]+equal_count
                perturbation[i] = perturbation[i] +  np.sum(np.abs(newimg-img))
                r.append(newimg[0])
            else:
                fake_count[i] = fake_count[i] + 1
        return np.array(r), cat_count,fake_count,perturbation

    def attack_single(self, nimg_set, img, target):
        """
        Run the attack on a single image and label
        """
        prev = np.copy(nimg_set).reshape(
            (BATCH_SIZE, self.model.image_size, self.model.image_size, self.model.num_channels))

        # initially set the solution to None, if we can't find an adversarial
        # example then we will return None as the solution.
        Equal_count, nimg = self.grad(np.copy(img), target, np.copy(prev))
        return Equal_count, nimg

