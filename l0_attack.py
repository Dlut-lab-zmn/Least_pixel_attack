## contained in the LICENCE file in this directory.
from __future__ import print_function


import tensorflow as tf
import numpy as np
from numpy import random
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
LEARNING_RATE = 1e-2    # larger values converge faster to less accurate results
TARGETED = True

class CarliniL0:
    def __init__(self, sess, model,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 max_iterations = MAX_ITERATIONS, abort_early = ABORT_EARLY,
                 independent_channels = False):
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
        def compare(x,y):
            if self.TARGETED:
                return x == y
            else:
                return x != y
        shape = (1,model.image_size,model.image_size,model.num_channels)
        
        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        # the variables we're going to hold, use for efficiency
        canchange = tf.Variable(np.zeros(shape),dtype=np.float32)
        simg = tf.Variable(np.zeros(shape,dtype=np.float32))
        original = tf.Variable(np.zeros(shape,dtype=np.float32))
        timg = tf.Variable(np.zeros(shape,dtype=np.float32))
        tlab = tf.Variable(np.zeros((1,model.num_labels),dtype=np.float32))


        # and the assignment to set the variables
        assign_modifier = tf.placeholder(np.float32,shape)
        assign_canchange = tf.placeholder(np.float32,shape)
        assign_simg = tf.placeholder(np.float32,shape)
        assign_original = tf.placeholder(np.float32,shape)
        assign_timg = tf.placeholder(np.float32,shape)
        assign_tlab = tf.placeholder(np.float32,(1,self.model.num_labels))

        # these are the variables to initialize when we run
        set_modifier = tf.assign(modifier, assign_modifier)
        setup = []
        setup.append(tf.assign(canchange, assign_canchange))
        setup.append(tf.assign(timg, assign_timg))
        setup.append(tf.assign(original, assign_original))
        setup.append(tf.assign(simg, assign_simg))
        setup.append(tf.assign(tlab, assign_tlab))
        
        newimg = (tf.tanh(modifier + simg)/2)*canchange+(1-canchange)*original
        newimg = tf.clip_by_value(newimg , -0.5, 0.5)
        output = model.predict(newimg)
        
        real = tf.reduce_sum((tlab)*output,1)
        other = tf.reduce_max((1-tlab)*output - (tlab*10000),1)
        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+.01)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+.01)

        # sum up the losses
        loss2 = tf.reduce_sum(tf.square(newimg-tf.tanh(timg)/2))
        loss = 10.*loss1+loss2
            
        outgrad = tf.gradients(loss, [modifier])[0]
        
        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier,canchange,simg,
                                                  original,timg,tlab]+new_vars)

        
        def doit(oimgs, labs, starts, valid):
            # convert to tanh-space
            imgs = np.arctanh(np.array(oimgs)*1.999999)
            starts = np.arctanh(np.array(starts)*1.999999)

            # initialize the variables
            sess.run(init)
            sess.run(setup, {assign_timg: imgs, 
                                    assign_tlab:labs, 
                                    assign_simg: starts, 
                                    assign_original: oimgs,
                                    assign_canchange: valid})

            for step in range(self.MAX_ITERATIONS):


                    # remember the old value
                    oldmodifier = self.sess.run(modifier)
                    #if step%(self.MAX_ITERATIONS//10) == 0:
                    #    print(step,*sess.run((loss1,loss2),feed_dict=feed_dict))

                    # perform the update step
                    _, works, scores = sess.run([train, loss1, output], feed_dict=feed_dict)

                    if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
                        if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                            if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                                raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
                    if works < .0001 and self.ABORT_EARLY:
                        # it worked previously, restore the old value and finish
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, output,newimg))
                        return grads, scores, nimg

        return doit
        
    def attack(self, imgs, targets,cat_count,perturbation,fake_count):
        """
        Perform the L_0 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        noise = (random.rand(self.model.image_size, self.model.image_size, self.model.num_channels)-0.5)/255.
        for i,(img,target) in enumerate(zip(imgs, targets)):
            #clip_imgae = np.around(np.floor(np.clip(( img + noise+0.5) * 255., 0., 255.)))
            #newimg = clip_imgae / 255. - 0.5
            oimg = img
            noise = (random.rand(self.model.image_size, self.model.image_size, self.model.num_channels) - 0.5)
            img = np.clip(img +noise, -0.5, 0.5)
            newimg,equal_count = self.attack_single(oimg,img,target)
            if equal_count is not None:
                cat_count[i] = cat_count[i]+equal_count
                perturbation[i] = perturbation[i] + np.sum(np.abs(newimg-oimg))
                r.extend(newimg)
            else:
                fake_count[i] = fake_count[i] + 1
        return np.array(r),cat_count,fake_count,perturbation

    def attack_single(self, oimg, img, target):
        """
        Run the attack on a single image and label
        """

        # the pixels we can change
        valid = np.ones((1,self.model.image_size,self.model.image_size,self.model.num_channels))

        # the previous image
        prev = np.copy(img).reshape((1,self.model.image_size,self.model.image_size,
                                     self.model.num_channels))

        # initially set the solution to None, if we can't find an adversarial
        # example then we will return None as the solution.
        last_solution = None

        equal_count = None
    
        while True:
            # try to solve given this valid map
            res = self.grad([np.copy(oimg)], [target], np.copy(prev),valid)
            if res == None:
                # the attack failed, we return this as our final answer
                if equal_count is None:
                    return img,None
                else:
                    return last_solution,equal_count
    
            # the attack succeeded, now we pick new pixels to set to 0
            restarted = False
            gradientnorm, scores, nimg = res

            equal_count = np.sum(nimg!=oimg)#self.model.image_size**2-np.sum(np.all(np.abs(nimg-oimg)<.004,axis=2))

            #print("Forced equal:",np.sum(1-valid),"Equal count:",equal_count)
            if np.sum(valid) == 0:
                # if no pixels changed, return 
                return img,None
    
            if self.independent_channels:
                # we are allowed to change each channel independently
                valid = valid.flatten()
                totalchange = abs(nimg[0]-img)*np.abs(gradientnorm[0])
            else:
                # we care only about which pixels change, not channels independently
                # compute total change as sum of change for each channel
                valid = valid.reshape((self.model.image_size**2,self.model.num_channels))
                totalchange = abs(np.sum(nimg[0]-img,axis=2))*np.sum(np.abs(gradientnorm[0]),axis=2)
            totalchange = totalchange.flatten()

            # set some of the pixels to 0 depending on their total change
            did = 0
            for e in np.argsort(totalchange):
                if np.all(valid[e]):
                    did += 1
                    valid[e] = 0
                    if totalchange[e] > .01:
                        # if this pixel changed a lot, skip
                        break
                    if did >= .3*equal_count**.5:
                        # if we changed too many pixels, skip
                        break

            valid = np.reshape(valid,(1,self.model.image_size,self.model.image_size,-1))
    
            last_solution = prev = nimg
