import torchvision.models as models
import torch.nn as nn
from torchvision.models import SwinTransformer

MODEL_SIZES = ["tiny", "small", "base"]
MODEL_VERSIONS = ["v1", "v2"]


def swin_classifier(size: str, version: str, num_classes: int) -> SwinTransformer:
    """
    Builds a flexible Swin Transformer model for multi-class classification.

    Args:
        size: 'tiny', 'small', or 'base'
        version: 'v1' or 'v2'
        num_classes: Number of final output categories (N)
    """
    size = size.lower().strip()
    version = version.lower().strip()

    # Define valid string mappings to torchvision builders
    model_mapping = {
        "v1": {
            "tiny": (models.swin_t, models.Swin_T_Weights.DEFAULT),
            "small": (models.swin_s, models.Swin_S_Weights.DEFAULT),
            "base": (models.swin_b, models.Swin_B_Weights.DEFAULT),
        },
        "v2": {
            "tiny": (models.swin_v2_t, models.Swin_V2_T_Weights.DEFAULT),
            "small": (models.swin_v2_s, models.Swin_V2_S_Weights.DEFAULT),
            "base": (models.swin_v2_b, models.Swin_V2_B_Weights.DEFAULT),
        }
    }

    # Validate function inputs
    if version not in model_mapping:
        raise ValueError(f"Invalid model version: {version}. Choose version from {MODEL_VERSIONS}.")

    if size not in model_mapping[version]:
        raise ValueError(f"Invalid model size. Choose size from {MODEL_SIZES}.")

    # Get a reference to the model and its weights
    model_instance, model_weights = model_mapping[version][size]

    # Load weights first
    model = model_instance(weights=model_weights)

    # Rearrange the final head to map the desired output
    num_features = model.head.in_features
    model.head = nn.Linear(in_features=num_features, out_features=num_classes)

    return model


if __name__ == "__main__":
    my_model = swin_classifier(size="base", version="v2", num_classes=3)
    print(my_model)
