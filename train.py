from chainer.datasets import get_mnist
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.training.trainer import Trainer
from chainer.training.updater import StandardUpdater
from chainer.training.extensions import LogReport, PrintReport, snapshot_object
from iaf.model import create_sample_model


def main():
    # data
    train, test = get_mnist(withlabel=False)
    n_x = train.shape[1]

    # model
    model = create_sample_model(n_x)

    n_batch = 256
    train_iter = SerialIterator(train, n_batch)
    # TODO: report test loss
    # test_iter = SerialIterator(test, n_batch)

    optimizer = Adam()
    optimizer.setup(model)
    gpu = 0
    updater = StandardUpdater(train_iter, optimizer, device=gpu)

    n_epoch = 50
    trainer = Trainer(updater, (n_epoch, 'epoch'))
    trainer.extend(
        snapshot_object(
            model, filename='snapshot_epoch_{.updater.epoch:03d}.npz'),
        trigger=(1, 'epoch'))
    trainer.extend(LogReport())
    trainer.extend(PrintReport([
        'epoch', 'main/loss', 'main/iaf_loss', 'main/rec_loss',
    ]))

    trainer.run()


if __name__ == '__main__':
    main()
