
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "tiff", "tif"]

IMAGES_DIR_NAME = "Images"
GLI_DIR_NAME = "glioma"
MEN_DIR_NAME = "meningioma"
PIT_DIR_NAME = "pituitary"
META_FILE_NAME = "metadata.json"

IMAGES_PER_TUMOR_TYPE = 700
MAX_IMAGES_PER_TUMOR_TYPE = 708    # Limited by the smallest class (meningioma: 708 slices)

# Class indices must be zero-based and contiguous: they are used directly as CrossEntropyLoss
# targets and as indices into the model output layer.
LABELS = {"gli": 0, "men": 1, "pit": 2}
INV_LABELS = {0: "glioma", 1: "meningioma", 2: "pituitary"}

# Class names ordered by class index. This is the mapping from model outputs back to tumor types.
ORDERED_LABELS = [INV_LABELS[index] for index in range(len(INV_LABELS))]

MODEL_SIZES = ["tiny", "small", "base"]
MODEL_VERSIONS = ["v1", "v2"]

RANDOM_SEED = 42
KEYWORD = "swin"
RESULTS_FILE_NAME = "learn_curves.csv"
RUN_META_FILE_NAME = "run_metadata.json"
EPOCHS = 10
LRATE = 0.0001
BATCH = 16

IMAGE_SIZE_FOR_NN = 256
CROP_SIZE_FOR_NN = 224
CROP_SCALE = (0.7, 1.0)    # Area range kept by RandomResizedCrop. MRI slices do not survive harsh crops.
NUM_WORKERS = 4
