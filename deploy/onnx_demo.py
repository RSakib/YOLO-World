import os
import json
import argparse
import os.path as osp

import cv2
import numpy as np
import supervision as sv
import onnxruntime as ort
from mmengine.utils import ProgressBar

try:
    import torch
    from torchvision.ops import nms
except Exception as e:
    print(e)

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser('YOLO-World ONNX Demo')
    parser.add_argument('onnx', help='onnx file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help=
        'detecting texts (str or json), should be consistent with the ONNX model'
    )
    parser.add_argument('--output-dir',
                        default='./output',
                        help='directory to save output files')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference')
    parser.add_argument(
        '--onnx-nms',
        action='store_false',
        help='whether ONNX model contains NMS and postprocessing')
    args = parser.parse_args()
    return args


def preprocess(image, size=(640, 640)):
    h, w = image.shape[:2]
    max_size = max(h, w)
    scale_factor = size[0] / max_size
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    pad_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)
    pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image
    image = cv2.resize(pad_image, size,
                       interpolation=cv2.INTER_LINEAR).astype('float32')
    image /= 255.0
    image = image[None]
    return image, scale_factor, (pad_h, pad_w)


def visualize(image, bboxes, labels, scores, texts):
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image


def inference(ort_session,
              image_path,
              texts,
              output_dir,
              size=(640, 640),
              **kwargs):
    # normal export
    # with NMS and postprocessing
    ori_image = cv2.imread(image_path)
    h, w = ori_image.shape[:2]
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)
    input_ort = ort.OrtValue.ortvalue_from_numpy(image.transpose((0, 3, 1, 2)))
    results = ort_session.run(["num_dets", "labels", "scores", "boxes"],
                              {"images": input_ort})
    num_dets, labels, scores, bboxes = results
    num_dets = num_dets[0][0]
    labels = labels[0, :num_dets]
    scores = scores[0, :num_dets]
    bboxes = bboxes[0, :num_dets]

    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    bboxes /= scale_factor
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
    bboxes = bboxes.round().astype('int')

    
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        image_height, image_width, image_channels = ori_image.shape
        box_xmin, box_ymin, box_xmax, box_ymax = bbox

        padding = 20
        box_xmin -= padding
        box_ymin -= padding
        box_xmax += padding
        box_ymax += padding

        if box_xmin < 0:
            box_xmin = 0
                            
        if box_ymin < 0:
            box_ymin = 0

        if box_xmax > image_width:
            box_xmax = image_width

        if box_ymax > image_height:
            box_ymax = image_height

                            
        crop = ori_image[box_ymin:box_ymax, box_xmin:box_xmax,:]

        fileName, fileExt = os.path.splitext(osp.basename(image_path))

        cv2.imwrite(osp.join(output_dir, f'{fileName}crop{str(i+1)}{fileExt}'), crop)

    image_out = visualize(ori_image, bboxes, labels, scores, texts)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)

    return image_out


def inference_with_postprocessing(ort_session,
                                  image_path,
                                  texts,
                                  output_dir,
                                  size=(640, 640),
                                  nms_thr=0.65,
                                  score_thr=0.05,
                                  max_dets=300):
    # export with `--without-nms`
    ori_image = cv2.imread(image_path)
    h, w = ori_image.shape[:2]
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)
    input_ort = ort.OrtValue.ortvalue_from_numpy(image.transpose((0, 3, 1, 2)))
    results = ort_session.run(["scores", "boxes"], {"images": input_ort})
    scores, bboxes = results
    # move numpy array to torch
    ori_scores = torch.from_numpy(scores[0]).to('cuda:0')
    ori_bboxes = torch.from_numpy(bboxes[0]).to('cuda:0')

    scores, labels = torch.max(ori_scores, dim=1)
    keep_idx = (scores > 0.001)
    bboxes = ori_bboxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    # batched nms
    bbox_inds = batched_nms(bboxes, scores, labels, iou_threshold=nms_thr)

    scores = scores[bbox_inds]
    bboxes = bboxes[bbox_inds]
    labels = labels[bbox_inds]

    scores_list = []
    labels_list = []
    bboxes_list = []

    if bbox_inds.shape[0] > max_dets:
        for cls_id in range(len(texts)):
            scores_cls = scores[labels == cls_id]
            print(scores_cls.shape)
            if scores_cls.shape[0] == 0:
                continue
            _, index = scores_cls.sort()
            keep_inds = index[:max_dets]
            box_cls = bboxes[labels == cls_id][keep_inds]
            scores_cls = scores[labels == cls_id][keep_inds]
            labels_cls = labels[labels == cls_id][keep_inds]

            scores_list.append(scores_cls)
            labels_list.append(labels_cls)
            bboxes_list.append(box_cls)
        scores = torch.cat(scores_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        bboxes = torch.cat(bboxes_list, dim=0)


    keep_idxs = scores > score_thr
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]

    # Get candidate predict info by num_dets
    scores = scores.cpu().numpy()
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()

    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    bboxes /= scale_factor
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
    bboxes = bboxes.round().astype('int')

    image_out = visualize(ori_image, bboxes, labels, scores, texts)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)
    return image_out


def main():

    args = parse_args()
    onnx_file = args.onnx
    # init ONNX session
    ort_session = ort.InferenceSession(
        onnx_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Init ONNX Runtime session")
    output_dir = "onnx_outputs"
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines]
    elif args.text.endswith('.json'):
        texts = json.load(open(args.text))
    else:
        texts = [[t.strip()] for t in args.text.split(',')]

    print("Start to inference.")
    progress_bar = ProgressBar(len(images))

    if args.onnx_nms:
        inference_func = inference
    else:
        inference_func = inference_with_postprocessing

    for img in images:
        inference_func(ort_session, img, texts, output_dir=output_dir)
        progress_bar.update()
    print("Finish inference")


if __name__ == "__main__":
    main()
