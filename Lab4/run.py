import numpy as np

from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28, 28]
    # n_train = 60000
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=1000)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")

    BATCH_SIZE = 10
    EPOCHS = 20
    RBM = False
    PLOT_ERRORS = False

    if RBM:
        rec_errs = {}
        for n_hidden in [500]:
            rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                             ndim_hidden=n_hidden,
                                             is_bottom=True,
                                             image_size=image_size,
                                             is_top=False,
                                             n_labels=10,
                                             batch_size=BATCH_SIZE
            )

            rbm.cd1(visible_trainset=train_imgs, n_iterations=EPOCHS)
            rec_errs[n_hidden] = rbm.reconstruction_err

        if PLOT_ERRORS:
            plt.title('Reconstruction loss for different numbers of hidden units')
            plt.xlabel('Epoch')
            plt.ylabel('Reconstruction loss')
            plt.plot([int(x) + 1 for x in np.arange(len(rec_errs[200]))], rec_errs[200], label='200hn')
            plt.plot([int(x) + 1 for x in np.arange(len(rec_errs[500]))], rec_errs[500], label='500hn')
            plt.legend()
            plt.show()

    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=EPOCHS)
    dbn.recognize(train_imgs, train_lbls)
    dbn.recognize(test_imgs, test_lbls)

    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0, digit] = 1
    #     dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)
    #
    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
