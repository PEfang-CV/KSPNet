import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CIS_SEM_SEG_CATEGORIES = [
    "void", "inclusion", "sundry", "red-iron sheet", "abrasion mask", "liquid", "roll printing", "water spot", "oxide scale", "scratch", "oil spot", "rail defect,", "patch",
]


def register_CID(root):
    root = os.path.join(root, "CID-Seg")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations", dirname)
        name = f"CID_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=CIS_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_CID(_root)