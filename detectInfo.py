import math
import sys,os,glob
from pathlib import Path
import cv2
import torch
import numpy as np
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
from utils.plots import colors, plot_one_box
class Detect_img:
    def __init__(self, weights, device=''):
        # Load model
        set_logging()
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model

    @torch.no_grad()
    def get_info(self,
            source='data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            save_img = False,
            save_dir = "",
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            half=False,  # use FP16 half-precision inference
            ):

        # Directories
        os.makedirs(save_dir,exist_ok=True)
        # (save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Initialize
        device = self.device
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        model = self.model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Set Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        batch_imgbox_list = []  # 存储批量图像的box信息
        batch_small_circle_list = []
        batch_large_circle_list = []
        for path, img, im0s, vid_cap in dataset: #一个batch的信息
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_synchronized()

            # Process detections
            imgbox_list = [] #一张图的box信息
            small_circle_list = []
            large_circle_list = []
            for i, det in enumerate(pred):  # detections per image 一张图信息
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % img.shape[2:]  # print string
                p = Path(p)  # to Path
                save_path = save_dir+p.name # img.jpg
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                    # 此时坐标格式为xyxy
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # for *xyxy, conf, cls in reversed(det):
                    for *xyxy, conf, cls in det: #置信度从大到小 #图中的所有框信息
                        xywh_no_norm = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                        box = (cls, xywh_no_norm, conf) #存储类别 xywh 置信度
                        if(cls==0):
                            small_circle_list.append(box)
                        if(cls==1):
                            large_circle_list.append(box)
                        imgbox_list.append(box)
                        c = int(cls)  # integer class
                        label =  f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            batch_imgbox_list.append(imgbox_list)
            batch_small_circle_list.append(small_circle_list)
            batch_large_circle_list.append(large_circle_list)

        return  batch_imgbox_list,batch_small_circle_list,batch_large_circle_list

def adjust_img_list(batch_small_circle_list, batch_large_circle_list, img_path_list, width, height, save_img=False, save_dir="",padding=150):
    def find_circle(small_circle_list, large_circle_list):
        center_xy = [large_circle_list[0][1][0],large_circle_list[0][1][1]]
        dis_list = []
        for box in small_circle_list:
            xy = [box[1][0],box[1][1]]
            dis = math.sqrt((center_xy[0]-xy[0])**2+(center_xy[1]-xy[1])**2)
            dis_list.append([xy,dis])
        dis_list = sorted(dis_list,key=lambda x:(x[1]),reverse=True)
        ratio = dis_list[0][1]/dis_list[1][1]
        return center_xy,dis_list[0][0],dis_list[1][0],ratio

    Rimg_list = []
    for i,img_path in enumerate(img_path_list):
        img = cv2.imread(img_path)
        small_circle_list = batch_small_circle_list[i]
        large_circle_list = batch_large_circle_list[i]
        if not(len(small_circle_list) == 4 and len(large_circle_list) == 1) :continue
        assert len(small_circle_list)==4 and len(large_circle_list)==1
        #get points
        center_xy,long_circle_xy,short_circle_xy,ratio = find_circle(small_circle_list,large_circle_list)
        # revise img
        scr_xy = [center_xy,long_circle_xy,short_circle_xy]
        # print(scr_xy)
        dst_xy = [[padding,padding],[padding,padding+ratio*(width-2*padding)],[width-padding,padding]]
        scr_xy = np.float32(scr_xy)
        dst_xy = np.float32(dst_xy)
        M = cv2.getAffineTransform(scr_xy, dst_xy)
        Rimg = cv2.warpAffine(img, M, (width, height))
        p = Path(img_path)
        # print(p.name)
        #save result
        if(save_img):
            cv2.imwrite(save_dir+p.name,img)
            cv2.imwrite(save_dir+p.name[:-4]+"_revice.jpg", Rimg)
        Rimg_list.append(Rimg)

    return Rimg_list

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  #libiomp5md.dll,有多个
    Detect_img = Detect_img(weights="./runs/train/exp5/weights/best.pt")
    img_root = "./data/temp_result2/temp"
    save_dir_anno = "./data/temp_result2/anno/"
    save_dir_img = "./data/temp_result2/"
    batch_imgbox_list,batch_small_circle_list,batch_large_circle_list = Detect_img.get_info(source=img_root,save_img=True,save_dir=save_dir_anno)
    img_path_list = [path for path in glob.glob(img_root+"/*.jpg")]
    # img_path_list = ["./data/Dataset/images/test/7_O_63557361925909330880.jpg"]

    Rimg_list = adjust_img_list(batch_small_circle_list, batch_large_circle_list, img_path_list,save_dir=save_dir_img,width=1800,height=3300,save_img=True)
    # print(batch_imgbox_list)
    # print(batch_small_circle_list)
    # print(batch_large_circle_list)

