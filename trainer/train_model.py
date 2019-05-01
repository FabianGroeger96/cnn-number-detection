from time import sleep
from trainer import Trainer


def main():
    trainer = Trainer()

    #trainer.create_gnet_model_deep('with-random-images')
    #trainer.fit_model()
    #trainer.save_model()

    trainer.create_gnet_model_light('with-random-images')
    trainer.fit_model()
    trainer.save_model()

    #trainer.create_various_models()


if __name__ == "__main__":
    main()
