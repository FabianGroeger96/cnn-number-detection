from trainer.model_gnet_deep import ModelGNetDeep
from trainer.model_gnet_light import ModelGNetLight
from trainer.model_gnet_light_v2 import ModelGNetLightV2
from trainer.model_various import ModelVarious


def main():

    model = ModelGNetLight('lego-dataset')
    model.train_model()
    model.save_model()


if __name__ == "__main__":
    main()
