import torch
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms
import numpy as np
import os
from collections.abc import Iterable


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    if not isinstance(imgs, Iterable):
        imgs = [imgs]
    if any(img.size != imgs[0].size for img in imgs):
        raise Exception("MTCNN batch processing only compatible with equal-dimension images.")

    imgs = [torch.as_tensor(np.uint8(img)).float().to(device) for img in imgs]
    imgs = torch.stack(imgs).permute(0, 3, 1, 2)

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage
    boxes = []
    image_inds = []
    all_i = 0
    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)
    
        box_inds_scale = []
        for b_i in range(batch_size):
            boxes_i = generateBoundingBox(reg[b_i], probs[b_i, 1], scale, threshold[0])
            boxes.append(boxes_i)
            
            image_inds.extend([[b_i, all_i] for _ in range(len(boxes_i))])
            all_i += 1

    boxes = torch.cat(boxes, axis=0)
    image_inds = torch.as_tensor(image_inds)

    # NMS within each scale + image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds[:, 1], 0.5)
    boxes, image_inds = boxes[pick], image_inds[pick]
    
    # NMS within each image
    image_inds = image_inds[:, 0]
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)
    boxes[:, :4] = boxes[:, :4].trunc().int()
    y, ey, x, ex = pad(boxes, w, h)
    
    # Second stage
    im_data = []
    for k in range(len(y)):
        if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
            img_k = imgs[image_inds[k]][:, (y[k] - 1) : ey[k], (x[k] - 1) : ex[k]].unsqueeze(0)
            im_data.append(imresample(img_k, (24, 24)))
    im_data = torch.cat(im_data, axis=0)
    im_data = (im_data - 127.5) * 0.0078125
    out = rnet(im_data)

    out0 = out[0].permute(1, 0)
    out1 = out[1].permute(1, 0)
    score = out1[1, :]
    ipass = score > threshold[1]
    boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), axis=1)
    image_inds = image_inds[ipass]
    mv = out0[:, ipass].permute(1, 0)

    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
    total_boxes = bbreg(boxes, mv)
    total_boxes = rerec(boxes)

    boxes = boxes.cpu().numpy()
    total_boxes_all = []
    for b_i in range(batch_size):
        b_i_inds = image_inds == b_i
        total_boxes_all.append(boxes[b_i_inds])

    batch_boxes = []
    batch_points = []
    for img, total_boxes in zip(imgs, total_boxes_all):
        points = np.zeros((2, 5, 0))
        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            total_boxes = np.fix(total_boxes).astype(np.int32)
            y, ey, x, ex = pad(total_boxes.copy(), w, h)
            im_data = []
            for k in range(0, numbox):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[:, (y[k] - 1) : ey[k], (x[k] - 1) : ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (48, 48)))
            im_data = torch.cat(im_data, 0)
            im_data = (im_data - 127.5) * 0.0078125
            out = onet(im_data)

            out0 = np.transpose(out[0].cpu().numpy())
            out1 = np.transpose(out[1].cpu().numpy())
            out2 = np.transpose(out[2].cpu().numpy())
            score = out2[1, :]
            points = out1
            ipass = np.where(score > threshold[2])
            points = points[:, ipass[0]]
            total_boxes = np.hstack(
                [total_boxes[ipass[0], :4].copy(), np.expand_dims(score[ipass].copy(), 1)]
            )
            mv = out0[:, ipass[0]]

            w_i = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h_i = total_boxes[:, 3] - total_boxes[:, 1] + 1
            points_x = (
                np.tile(w_i, (5, 1)) * points[:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
            )
            points_y = (
                np.tile(h_i, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
            )
            points = np.stack((points_x, points_y), axis=0)
            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, np.transpose(mv))
                pick = nms(total_boxes, 0.7, "Min")
                total_boxes = total_boxes[pick, :]
                points = points[:, :, pick]

        batch_boxes.append(total_boxes[:, :5])
        batch_points.append(np.transpose(points))

    return np.array(batch_boxes), np.array(batch_points)


def bbreg(boundingbox, reg):

    try:
        if reg.shape[1] == 1:
            reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    except:
        if reg.shape[1] == 1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        boundingbox[:, :4] = np.transpose(np.vstack([b1, b2, b3, b4]))

    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    mask = probs >= thresh
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask.nonzero().float().flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox


def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[:counter]
    return pick


def pad(boxes, w, h):
    
    try:
        boxes = boxes.int()
        x = boxes[:, 0]
        y = boxes[:, 1]
        ex = boxes[:, 2]
        ey = boxes[:, 3]

        x[x < 1] = 1
        y[y < 1] = 1
        ex[ex > w] = w
        ey[ey > h] = h

    except:
        x = boxes[:, 0].copy().astype(np.int32)
        y = boxes[:, 1].copy().astype(np.int32)
        ex = boxes[:, 2].copy().astype(np.int32)
        ey = boxes[:, 3].copy().astype(np.int32)

        x[np.where(x < 1)] = 1
        y[np.where(y < 1)] = 1
        ex[np.where(ex > w)] = w
        ey[np.where(ey > h)] = h

    return y, ey, x, ex


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    
    try:
        l = torch.max(w, h)
        bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
        bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
        bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

    except:
        l = np.maximum(w, h)
        bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
        bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
        bboxA[:, 2:4] = bboxA[:, :2] + np.transpose(np.tile(l, (2, 1)))

    return bboxA


def imresample(img, sz):
    out_shape = (sz[0], sz[1])
    im_data = torch.nn.functional.interpolate(img, size=out_shape, mode="area")
    return im_data


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img.size[0])),
        int(min(box[3] + margin[1] / 2, img.size[1])),
    ]

    face = img.crop(box).resize((image_size, image_size), 2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_args = {"compress_level": 0} if ".png" in save_path else {}
        face.save(save_path, **save_args)

    face = F.to_tensor(np.float32(face))

    return face
