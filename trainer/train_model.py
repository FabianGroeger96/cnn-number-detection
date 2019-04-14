from trainer import Trainer


def main():
    trainer = Trainer()

    trainer.create_model()
    trainer.fit_model()
    trainer.save_model()

if __name__ == "__main__":
    main()