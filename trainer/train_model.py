from time import sleep
from trainer import Trainer


def main():
    trainer = Trainer()

    trainer.create_model_deep()
    trainer.fit_model()
    trainer.save_model()
    trainer.convert_model_tensorflow()

    # trainer.create_model_light()
    # trainer.fit_model()
    # trainer.save_model()

    #trainer.create_various_models()


if __name__ == "__main__":
    main()
