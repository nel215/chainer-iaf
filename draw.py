import os
import re
import matplotlib
from chainer.datasets import get_mnist
from chainer.serializers import load_npz
from iaf.model import create_sample_model
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa


def main():
    # data
    _, test = get_mnist(withlabel=False)
    n_x = test.shape[1]

    # model
    model = create_sample_model(n_x)
    test = test.astype('f')[:25]

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(test[i].reshape(28, 28), cmap='gray_r')
    plt.savefig('./result/ans.png')
    plt.close()

    pattern = re.compile(r'.+npz$')
    for fname in sorted(os.listdir('./result')):
        if not pattern.match(fname):
            continue
        out = './result/{}.png'.format(fname)
        if os.path.exists(out):
            continue

        print(fname)
        load_npz(os.path.join('./result', fname), model)

        gen_x = model.generate(test)
        gen_x = gen_x.reshape(-1, 28, 28)
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.imshow(gen_x[i], cmap='gray_r')
        plt.savefig(out)
        plt.close()


if __name__ == '__main__':
    main()
