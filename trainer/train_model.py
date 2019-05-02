from trainer.models.model_gnet_light import ModelGNetLight


def main():

    model = ModelGNetLight('lego-dataset')
    model.train_model()
    model.save_model()


if __name__ == "__main__":
    main()
