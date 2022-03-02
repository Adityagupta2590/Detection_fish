
def img_clg(data_pth):
    import cv2



    import torch
    assert torch.__version__.startswith("1.8") 
    import torchvision
    import cv2
    import numpy as np
    import cv2
    import os
    import numpy as np
    import json
    import random
    import matplotlib.pyplot as plt
     
    from detectron2.structures import BoxMode
    from detectron2.data import DatasetCatalog, MetadataCatalog

    fn =data_pth 
    print('path is',fn)

    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()
    
    # import some common libraries
    import numpy as np
    import cv2
    import random
    
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.utils.visualizer import ColorMode
    from detectron2.data import MetadataCatalog
    #from detectron2.utils import *
    #from detectron2.evaluation import *

    classes = ['fish']
    data_path = "test_fish/"
    for d in ["train", "test"]:
        DatasetCatalog.register("category_" + d, lambda d=d:get_data_dicts(data_path+d, classes))
        MetadataCatalog.get("category_" + d).set(thing_classes=classes)
    microcontroller_metadata = MetadataCatalog.get("category_train")
    
    def get_data_dicts(directory, classes):
        dataset_dicts = []
        for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
            json_file = os.path.join(directory, filename)
            with open(json_file) as f:
                img_anns = json.load(f)

            record = {}
        
            filename = os.path.join(directory, img_anns["imagePath"])
        
            record["file_name"] = filename
            record["height"] = 736
            record["width"] = 1920
      
            annos = img_anns["shapes"]
            objs = []
            for anno in annos:
                px = [a[0] for a in anno['points']] # x coord
                py = [a[1] for a in anno['points']] # y-coord
                poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(anno['label']),
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts
    
    #import glob
    import os
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    classes = ['fish']
    data_path = 'test_fish/'
    
    from detectron2.utils.visualizer import ColorMode, Visualizer
    # Parameters of detectron2
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu' ########to run in cpu
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("category_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.DATASETS.TEST = ("fish segmenttation", )

    predictor = DefaultPredictor(cfg)
        
    OUTS = []
    
    img = cv2.imread(fn)
    dim=[1920,736]
    img=cv2.resize(img,dim)
    print('resized image size is', img.shape)
    outputs = predictor(img)
    OUTS.append(outputs['instances'])
    #v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    #v = v.draw_instance_predictions(outputs['instances'].to("cpu"))    
    #cv2.imwrite('anotate.jpg', v.get_image()[:, :, ::-1])
    boxes = outputs["instances"].pred_boxes
    x = list(boxes.tensor.shape)
    print('no of fish is',x[0])
    ######################################################
    mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
    num_instances = mask_array.shape[0]
    scores = outputs['instances'].scores.to("cpu").numpy()
    labels = outputs['instances'].pred_classes .to("cpu").numpy()
    bbox   = outputs['instances'].pred_boxes.to("cpu").tensor.numpy()

    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_array_instance = []
        #img = np.zeros_like(im) #black
    h = img.shape[0]
    w = img.shape[1]
    img_mask = np.zeros([h, w, 3], np.uint8)
    color = (255, 255, 255)
   
    for i in range(num_instances):
        img = np.zeros_like(img)
        mask_array_instance.append(mask_array[:, :, i:(i+1)])
        img = np.where(mask_array_instance[i] == True, 255, img)
        array_img = np.asarray(img)
        img_mask[np.where((array_img==[255,255,255]).all(axis=2))]=color    
    img_mask = np.asarray(img_mask)
    
    rgb_w=[0.29,0.58,0.11]
    gr_im=np.dot(img_mask[...,:3],rgb_w)
    im=np.array(gr_im)
    print("done")
    return(gr_im)


#from test2 import img_clg
#from test2 import *
import cv2
#pip install -r requirements.txt
data_pth='test1.jpg'
print('data_path is',data_pth)
seg_img=img_clg(data_pth)
cv2.imwrite('test_fish/fish_seg.jpg',seg_img)
print('the segmented image is stored inside test_fish folder')



