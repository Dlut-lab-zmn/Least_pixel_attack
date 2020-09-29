# Least_pixel_attack

### About

Corresponding code to the paper "Further Understanding of the Least Pixel Attack" by Mn Zhao and Bo Wang,2020.

Implementations of the BFs and BFM attack algorithms in Tensorflow. It runs correctly
on Python 3 and Python 2.

To evaluate the robustness of a neural network, create a model class with a
predict method that will run the prediction network *without softmax*.  The
model should have variables 

    model.image_size: size of the image (e.g., 28 for MNIST, 32 for CIFAR)
    model.num_channels: 1 for greyscale, 3 for color images
    model.num_labels: total number of valid labels (e.g., 10 for MNIST/CIFAR)

### Running attacks

```python
     python test_attack.py
```
One can easily change each attack method in this file.

    #attack = CarliniL0_batch(sess, model, max_iterations=1000)
    #attack = CarliniL0(sess, model, max_iterations=1000)
    #attack = CarliniL0_batch(sess, model, max_iterations=1000)
    #attack = LPA_attack(sess, model, max_iterations=2000)
    #attack = LPA_attack2(sess, model, max_iterations=2000)
    #attack = LPA_attack_batch(sess, model, max_iterations=2000)

   
#### To create the MNIST/CIFAR models:

```bash
python train_models.py
```

#### And finally to test the attacks

```bash
python3 test_attack.py
```

