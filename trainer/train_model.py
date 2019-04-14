from time import sleep
from trainer import Trainer


def main():
    trainer = Trainer()

    print('[INFO] creating model')
    sleep(.5)
    trainer.create_model()

    print('[INFO] training model')
    sleep(.5)
    trainer.fit_model()

    print('[INFO] saving model')
    sleep(.5)
    trainer.save_model()


if __name__ == "__main__":
    main()
