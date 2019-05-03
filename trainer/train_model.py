from trainer.models.model_gnet_light import ModelGNetLight


def main():

    model = ModelGNetLight('full-dataset-aug-rand-batch-16-epoch-10')
    model.train_model()
    model.save_model()


if __name__ == "__main__":
    main()
