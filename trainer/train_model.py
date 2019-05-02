from time import sleep
from model_gnet_deep import ModelGNetDeep
from model_gnet_light import ModelGNetLight
from model_gnet_light_v2 import ModelGNetLightV2
from model_various import ModelVarious


def main():

    model = ModelGNetLight()
    model.create_model('test-only')
    model.save_model()


if __name__ == "__main__":
    main()
