import torchvision
import timm
import torchvision.models as models
def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


def get_vit(name, pretrained=False):
    vit_models = {
        "vit_base_patch16_224": "vit_base_patch16_224",
        "vit_large_patch16_224": "vit_large_patch16_224",
    }
    if name not in vit_models.keys():
        raise KeyError(f"{name} is not a valid ViT version")
    # 按需加载 Vision Transformer 模型
    return timm.create_model(vit_models[name], pretrained=pretrained)


def get_efficientnet(name, pretrained=False):
    efficientnet_models = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_b3": models.efficientnet_b3,
    }

    if name not in efficientnet_models.keys():
        raise KeyError(f"{name} is not a valid EfficientNet version")

    # 按需加载 EfficientNet 模型
    return efficientnet_models[name](pretrained=pretrained)

def get_resnext(name, pretrained=False):
    resnext_models = {
        "resnext50_32x4d": torchvision.models.resnext50_32x4d,
        "resnext101_32x8d": torchvision.models.resnext101_32x8d,
    }

    if name not in resnext_models.keys():
        raise KeyError(f"{name} is not a valid ResNeXt version")

    # 按需加载 ResNeXt 模型
    return resnext_models[name](pretrained=pretrained)

#
# resnet_model = get_resnext("resnext101_32x8d", pretrained=False)
# n_features_resnet = resnet_model.fc.in_features  # 获取特征维度
# print(f"ResNet 特征维度: {n_features_resnet}")
#
# vit_model = get_vit("vit_large_patch16_224", pretrained=False)
# n_features_vit = vit_model.head.in_features  # 获取特征维度
# print(f"ViT 特征维度: {n_features_vit}")

#
