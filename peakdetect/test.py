from __future__ import division

import argparse
from tqdm.notebook import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from peakdetect.models import load_model
from peakdetect.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info, rescale_boxes
from peakdetect.utils.transforms import DEFAULT_TRANSFORMS
from peakdetect.utils.parse_config import parse_data_config


def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.
    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


# def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
#     """Evaluate model on validation dataset.
#     :param model: Model to evaluate
#     :type model: models.Darknet
#     :param dataloader: Dataloader provides the batches of images with targets
#     :type dataloader: DataLoader
#     :param class_names: List of class names
#     :type class_names: [str]
#     :param img_size: Size of each image dimension for yolo
#     :type img_size: int
#     :param iou_thres: IOU threshold required to qualify as detected
#     :type iou_thres: float
#     :param conf_thres: Object confidence threshold
#     :type conf_thres: float
#     :param nms_thres: IOU threshold for non-maximum suppression
#     :type nms_thres: float
#     :param verbose: If True, prints stats of model
#     :type verbose: bool
#     :return: Returns precision, recall, AP, f1, ap_class
#     """
#     model.eval()  # Set model to evaluation mode

#     Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

#     labels = []
#     sample_metrics = []  # List of tuples (TP, confs, pred)
#     with tqdm(dataloader, unit="batch", disable=False, leave=False) as tepoch:
#         for imgs, targets, _, _ in tepoch:
#             # Extract labels
#             labels += targets[:, 1].tolist()
#             # Rescale target
#             targets[:, 2:] = xywh2xyxy(targets[:, 2:])
#             targets[:, 2:] *= img_size

#             imgs = Variable(imgs.type(Tensor), requires_grad=False)

#             with torch.no_grad():
#                 outputs = model(imgs)
#                 outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

#             sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

#         if len(sample_metrics) == 0:  # No detections over whole validation set.
#             print("---- No detections over whole validation set ----")
#             return None

#         # Concatenate sample statistics
#         true_positives, pred_scores, pred_labels = [
#             np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
#         metrics_output = ap_per_class(
#             true_positives, pred_scores, pred_labels, labels)

#         print_eval_stats(metrics_output, class_names, verbose)

#     return metrics_output

def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.
    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    with tqdm(dataloader, unit="batch", disable=False, leave=False) as tepoch:
        for imgs, targets, _, _ in tepoch:
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size

            imgs = Variable(imgs.type(Tensor), requires_grad=False)

            with torch.no_grad():
                outputs = model(imgs)
                outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return None

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)

        # print_eval_stats(metrics_output, class_names, verbose)
        _plot_evaluation(imgs, targets, outputs,class_names)
        
    return metrics_output


def _plot_evaluation(imgs, targets, outputs, classes):
    imgs = imgs.to("cpu")
    targets = targets.to("cpu")

    img_ids = random.sample(range(imgs.size(0)),4)
    fig, axs = plt.subplots(2,4, figsize=(10,5), dpi=150)
    for i in range(2):
        for j in range(4):
            img_id = img_ids[j]
            axs[i,j].imshow(imgs[img_id].squeeze().numpy())

            # model prediction
            if i==0: 
                detections = outputs[img_id]
            # ground truth
            else:
                detections = targets[np.where(targets.numpy()[:,0]==img_id)]
                num_detections = detections.size(0)
                detections = torch.cat((detections, torch.ones(num_detections,1)),axis=1)
                detections = detections[:,[2,3,4,5,6,1]]

            detections = detections.to("cpu")
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            # Bounding-box colors
            cmap = plt.get_cmap("tab20b")
            colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
            bbox_colors = random.sample(colors, n_cls_preds)

            for x1, y1, x2, y2, conf, cls_pred in detections:

                # print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                axs[i,j].add_patch(bbox)
                # Add label
                axs[i,j].text(
                    x1,#-0.5*box_w,
                    y2,
                    fontsize=4,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox={"color": color, "pad": 0})
            axs[i,j].axis("off")
            axs[i,j].xaxis.set_major_locator(NullLocator())
            axs[i,j].yaxis.set_major_locator(NullLocator())

    fig.subplots_adjust(wspace=0.0, hspace=0.05)
    plt.show()

def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names

    precision, recall, AP, f1, ap_class = evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_names,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True)


if __name__ == "__main__":
    run()
