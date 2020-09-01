from generate_JSMA import generate_attacks


MNIST_SETS = ["mnist", "lenet-5-less-training", "mnist_defense_jsma", "mnist_defense_wjsma", "mnist_defense_tjsma"]
CIFAR10_SETS = ["cifar10", "cifar10_defense_jsma", "cifar10_defense_wjsma", "cifar10_defense_tjsma"]


def ATTACK(attack,dataset,first_index,settype, last_index, batch_size):
    """
    Applies the saliency map attack against the specified model.

    Parameters
    ----------
    model: str
        The name of the model used.
    attack: str
        The type of used attack (either "jsma", "wjsma" or "tjsma").
    set_type: str
        The type of set used (either "train" or "test").
    first_index:
        The index of the first image attacked.
    last_index: int
        The index of the last image attacked.
    batch_size: int
        The size of the image batches.
    """

    if dataset == 'mnist':
        from cleverhans.dataset import MNIST

        x_set, y_set = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000).get_set(settype)
        print(x_set.shape)
        gamma = 0.155
        file_path="/models/mnist"
    #elif model in CIFAR10_SETS:
    else:
        #from cleverhans.dataset import CIFAR10
        #x_set, y_set = CIFAR10(train_start=0, train_end=50000, test_start=0, test_end=10000).get_set(settype)
        #gamma = 0.155
        from setup_cifar import CIFAR
        data = CIFAR()
        x_set,y_set = data.test_data,data.test_labels
        print(x_set.shape)
        print(y_set)
        gamma = 0.155
        file_path="./Least_pixel_attack/models/cifar"
    #else:
    #    raise ValueError("Invalid model: " + model)

    generate_attacks(
        save_path="./Least_pixel_attack/models/data",
        file_path=file_path,
        dataset = dataset,
        x_set=x_set,
        y_set=y_set,
        attack=attack,
        gamma=gamma,
        first_index=first_index,
        last_index=last_index,
        batch_size=batch_size
    )
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default="attack")
    parser.add_argument('--model', type=str, default="cifar")
    parser.add_argument('--settype', type=str, default="test")
    parser.add_argument('--attack', type=str, default="wjsma")
    parser.add_argument('--firstindex', type=int, default=0)
    parser.add_argument('--lastindex', type=int, default=500)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--visual', type=str, default='single')
    args = parser.parse_args()

ATTACK( args.attack,args.model ,args.firstindex,args.settype, args.lastindex, args.batchsize)