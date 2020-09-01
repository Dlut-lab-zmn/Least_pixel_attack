## contained in the LICENCE file in this directory.
from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy import random

BATCH_SIZE = 30
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
LEARNING_RATE = 1e-2    # larger values converge faster to less accurate results
TARGETED = True

class BPs:
    def __init__(self, sess, model,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
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

    def gradient_descent(self, sess, model):
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y

        shape = (BATCH_SIZE, model.image_size, model.image_size, model.num_channels)

        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        # the variables we're going to hold, use for efficiency
        canchange = tf.Variable(np.zeros(shape), dtype=np.float32)
        canchange2 = tf.Variable(np.zeros(shape), dtype=np.float32)
        canchange3 = tf.Variable(np.zeros(shape), dtype=np.float32)
        simg = tf.Variable(np.zeros(shape, dtype=np.float32))
        original = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape, dtype=np.float32))
        tlab = tf.Variable(np.zeros((BATCH_SIZE, model.num_labels), dtype=np.float32))


        # and the assignment to set the variables
        assign_modifier = tf.placeholder(np.float32, shape)
        assign_canchange = tf.placeholder(np.float32, shape)
        assign_canchange2 = tf.placeholder(np.float32, shape)
        assign_canchange3 = tf.placeholder(np.float32, shape)
        assign_simg = tf.placeholder(np.float32, shape)
        assign_original = tf.placeholder(np.float32, shape)
        assign_timg = tf.placeholder(np.float32, shape)
        assign_tlab = tf.placeholder(np.float32, (BATCH_SIZE, self.model.num_labels))

        # these are the variables to initialize when we run
        set_modifier = tf.assign(modifier, assign_modifier)
        setup = []
        setup.append(tf.assign(canchange, assign_canchange))
        setup.append(tf.assign(canchange2, assign_canchange2))
        setup.append(tf.assign(canchange3, assign_canchange3))
        setup.append(tf.assign(timg, assign_timg))
        setup.append(tf.assign(original, assign_original))
        setup.append(tf.assign(simg, assign_simg))
        setup.append(tf.assign(tlab, assign_tlab))

        newimg = (tf.tanh(modifier + simg) / 2) * canchange + (1 - canchange) * original
        newimg2 = (tf.tanh(modifier + simg) / 2) * canchange2 + (1 - canchange2) * original
        newimg3 = (tf.tanh(modifier + simg) / 2) * canchange3 + (1 - canchange3) * original
        
        output = model.predict(newimg)
        output2 = model.predict(newimg2)
        output3 = model.predict(newimg3)
        
        real = tf.reduce_sum((tlab) * output, 1)
        real2 = tf.reduce_sum((tlab) * output2, 1)
        real3 = tf.reduce_sum((tlab) * output3, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)
        other2 = tf.reduce_max((1 - tlab) * output2 - (tlab * 10000), 1)
        other3 = tf.reduce_max((1 - tlab) * output3 - (tlab * 10000), 1)
        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + .01)
            loss2 = tf.maximum(0.0, other2 - real2 + .01)
            loss3 = tf.maximum(0.0, other3 - real3 + .01)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + .01)
            loss2 = tf.maximum(0.0, real2 - other2 + .01)
            loss3 = tf.maximum(0.0, real3 - other3 + .01)
        # sum up the losses
        loss4= tf.reduce_sum(tf.square(newimg - tf.tanh(timg) / 2), (1, 2, 3))
        loss5= tf.reduce_sum(tf.square(newimg2 - tf.tanh(timg) / 2), (1, 2, 3))
        loss6= tf.reduce_sum(tf.square(newimg3 - tf.tanh(timg) / 2), (1, 2, 3))
        loss = 10 * (loss1+loss3+loss2)/3. + (loss4+loss5+loss6)/3.

        outgrad = tf.gradients(loss, [modifier])[0]

        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier, canchange,canchange2, simg,
                                                  original, timg, tlab] + new_vars)

        def doit(oimgs, labs, starts, valid,valid2,valid3):
            # convert to tanh-space
            imgs = np.arctanh(np.array(oimgs) * 1.999999)
            starts = np.arctanh(np.array(starts) * 1.999999)
            # initialize the variables
            sess.run(init)
            sess.run(setup, {assign_timg: imgs,
                             assign_tlab: labs,
                             assign_simg: starts,
                             assign_original: oimgs,
                             assign_canchange: valid,
                             assign_canchange2: valid2,
                             assign_canchange3: valid3})


            for step in range(self.MAX_ITERATIONS):
                # remember the old value
                oldmodifier = self.sess.run(modifier)

                # if step%(self.MAX_ITERATIONS//10) == 0:
                #    print(step,*sess.run((loss1,loss2),feed_dict=feed_dict))

                # perform the update step
                if self.TARGETED:
                    _, works, ploss2,ploss3 = sess.run([train, loss1,loss2,loss3])
                    if np.any(np.array(works) < .001) and self.ABORT_EARLY:
                        # it worked previously, restore the old value and finish
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output, newimg))
                        #l2s = np.square(nimg - np.tanh(imgs) / 2).sum(axis=(1, 2, 3))
                        return grads, scores, nimg,valid
                    elif np.any(np.array(ploss2) < .001) and self.ABORT_EARLY:
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output2, newimg2))
                        return grads, scores, nimg,valid2
                    elif np.any(np.array(ploss3) < .001) and self.ABORT_EARLY:
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output3, newimg3))
                        return grads, scores, nimg,valid3
                else:
                    _, logit, logit1,logit2,gtlabel= sess.run([train, output, output2,output3,tlab], feed_dict=feed_dict)
                    if np.any(np.argmax(logit,1) != np.argmax(np.squeeze(gtlabel))):
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output, newimg))
                        return grads, scores, nimg,valid
                    elif np.any(np.argmax(logit1,1) != np.argmax(np.squeeze(gtlabel))):
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output, newimg))
                        return grads, scores, nimg,valid
                    elif np.any(np.argmax(logit2,1) != np.argmax(np.squeeze(gtlabel))):
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output, newimg))
                        return grads, scores, nimg,valid

        return doit

    def attack(self, imgs, targets,cat_count,perturbation,fake_count,targeted =False):
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
            #noise2 = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9]) - 0.5
            for b in range(BATCH_SIZE):
                noise = (random.rand(self.model.image_size, self.model.image_size, self.model.num_channels) - 0.5)# / (255.)
                # nimg_set.append(np.clip( img + noise2[b] / 255., -0.5, 0.5))
                nimg_set.append(np.clip(img +noise, -0.5, 0.5))#noise*edge[tar[i]]+noise 
                img_set.append(img)
                tar_set.append(target)
            newimg, equal_count = self.attack_single(nimg_set, img_set, tar_set)
            if equal_count is not None:
                if targeted:
                    index = np.argmax(self.model.model.predict(np.array(newimg)), 1) == i
                    cat_count[i] = cat_count[i]+np.max(index * equal_count)
                else:
                    index = np.argmax(self.model.model.predict(np.array(newimg)),1)!= target
                    cat_count[i] = cat_count[i]+np.max(index * equal_count)
                perturbation[i] = perturbation[i] + np.sum(np.abs(newimg[np.argmax(index * equal_count)]-img))
                r.append(newimg[np.argmax(index * equal_count)])
            else:
                fake_count[i] = fake_count[i] + 1
        return np.array(r),cat_count,fake_count,perturbation

    def attack_single(self, nimg_set, img, target):
        """
        Run the attack on a single image and label
        """

        # the pixels we can change
        #edge = np.repeat(np.expand_dims(edge,0),BATCH_SIZE,0)
        valid = np.ones((BATCH_SIZE, self.model.image_size, self.model.image_size, self.model.num_channels))#edge
        valid2 = np.ones((BATCH_SIZE, self.model.image_size, self.model.image_size, self.model.num_channels))#edge
        valid3 = np.ones((BATCH_SIZE, self.model.image_size, self.model.image_size, self.model.num_channels))#edge
        # the previous image
        prev = np.copy(nimg_set).reshape(
            (BATCH_SIZE, self.model.image_size, self.model.image_size, self.model.num_channels))
        
        # initially set the solution to None, if we can't find an adversarial
        # example then we will return None as the solution.
        last_solution = None
        equal_count = None

        while True:
            # try to solve given this valid map
            res = self.grad(np.copy(img), target, np.copy(prev),
                            valid,valid2, valid3)
            if res == None:
                # the attack failed, we return this as our final answer
                if equal_count is None:
                    return img, None
                else:
                    return last_solution, equal_count

            # the attack succeeded, now we pick new pixels to set to 0
            restarted = False
            gradientnorm, scores, nimg,valid = res

            cal_img = np.around(np.clip((np.array(img) + 0.5) * 255., 0., 255.))
            cal_nimg = np.around(np.clip((np.array(nimg) + 0.5) * 255., 0., 255.))
            equal_count = self.model.image_size ** 2 - np.sum(np.all(np.abs(cal_img - cal_nimg) < 1, axis=3), axis=(1, 2))
            #print("Forced equal:",np.sum(1-valid),"Equal count:",equal_count)
            if np.sum(valid) == 0:
                # if no pixels changed, return
                return img,None

            if self.independent_channels:
                # we are allowed to change each channel independently
                valid = valid.flatten()
                totalchange = abs(nimg - img) * np.abs(gradientnorm)
            else:
                # we care only about which pixels change, not channels independently
                # compute total change as sum of change for each channel
                valid = valid.reshape((BATCH_SIZE, self.model.image_size ** 2, self.model.num_channels))
                valid2 = valid.copy()
                valid3 = valid.copy()
                #totalchange = np.sum(np.abs(gradientnorm), axis=3)/(abs(np.sum(nimg - img, axis=3))+0.001)
                totalchange = np.sum(np.abs(gradientnorm), axis=3)*(abs(np.sum(nimg - img, axis=3)))
            # set some of the pixels to 0 depending on their total change
            for b in range(BATCH_SIZE):
                change_pixel = totalchange[b].flatten()
                did = 0
                did2 = 0
                did3 = 0
                for e in np.argsort(change_pixel):
                    if np.all(valid[b, e]):
                        did += 1
                        did2 += 1
                        did3 += 1
                        if did <= 2:
                            valid[b, e] = 0
                            valid2[b, e] = 0
                            valid3[b, e] = 0
                        elif did2 <=3:
                            valid2[b, e] = 0
                            valid3[b, e] = 0
                        elif did3 <= 5:
                            valid3[b, e] = 0
                        else:
                            break
            valid = np.reshape(valid, (BATCH_SIZE, self.model.image_size, self.model.image_size, -1))
            valid2 = np.reshape(valid2, (BATCH_SIZE, self.model.image_size, self.model.image_size, -1))
            valid3 = np.reshape(valid3, (BATCH_SIZE, self.model.image_size, self.model.image_size, -1))
            last_solution = prev = nimg
