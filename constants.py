
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "tiff", "tif"]

IMAGES_DIR_NAME = "Images"
GLI_DIR_NAME = "glioma"
MEN_DIR_NAME = "meningioma"
PIT_DIR_NAME = "pituitary"
META_FILE_NAME = "metadata.json"

IMAGES_PER_TUMOR_TYPE = 700
MAX_IMAGES_PER_TUMOR_TYPE = 704

LABELS = {"gli": 1, "men": 2, "pit": 3}
INV_LABELS = {1: "glioma", 2: "meningioma", 3: "pituitary"}

RANDOM_SEED = 42
