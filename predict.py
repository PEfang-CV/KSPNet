import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import cog
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

class Predictor(cog.Predictor):
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("configs/CDS/CID_R101.yaml")
        cfg.MODEL.WEIGHTS = 'XXX'
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        self.predictor = DefaultPredictor(cfg)
        self.coco_metadata = MetadataCatalog.get("CID_sem_seg_val")

    @cog.input(
        "image",
        type=Path,
        help="Input image for segmentation. Output will be the concatenation of Panoptic segmentation (top), "
             "instance segmentation (middle), and semantic segmentation (bottom).",
    )
    def predict(self, image):
        im = cv2.imread(str(image))
        outputs = self.predictor(im)
        sem_seg = outputs["sem_seg"].argmax(0).to("cpu").numpy()
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), sem_seg)
        return out_path
