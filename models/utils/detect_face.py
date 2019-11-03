import torch
import torchvision.transforms.functional as F
from torchvision.ops.boxes import nms
import numpy as np
import os
from collections.abc import Iterable


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    if not isinstance(imgs, Iterable):
        imgs = [imgs]
    if any(img.size != imgs[0].size for img in imgs):
        raise Exception('MTCNN batch processing only compatible with equal-dimension images.')

    imgs = [torch.as_tensor(np.uint8(img)).float().to(device) for img in imgs]
    imgs = torch.stack(imgs).permute(0, 3, 1, 2)

    h, w = imgs.shape[2:4]
    scale = 12.0 / minsize
    minl = min(h, w) * scale

    # Proposal network
    proposal_boxes = []
    while minl >= 12:
        hs = int(h * scale + 1)
        ws = int(w * scale + 1)
        img_x = imresample(imgs, (hs, ws))
        img_x = (img_x - 127.5) * 0.0078125
        reg, probs = pnet(img_x)
        
        for b_i in range(batch_size):
            boxes = generateBoundingBox(reg[b_i], probs[b_i, 1], scale, threshold[0])
            pick = nms(boxes[:, :4], boxes[:, 4], 0.5)
            boxes = boxes[pick]
            proposal_boxes.append(boxes)

        scale = scale * factor
        minl = minl * factor

    batch_boxes = []
    for img, total_boxes in zip(imgs, proposal_boxes):
        numbox = total_boxes.shape[0]
        if numbox>0:
            pick = nms(total_boxes[:, :4], total_boxes[:, 4], 0.7)
            total_boxes = total_boxes[pick]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
            total_boxes = torch.stack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]).permute(1, 0)
            total_boxes = rerec(total_boxes)
            total_boxes[:, 0:4] = total_boxes[:, 0:4].int()
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes, w, h)

        # Refinement network
        numbox = total_boxes.shape[0]
        if numbox > 0:
            im_data = []
            for k in range(0, numbox):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[:, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (24, 24)))
            im_data = torch.cat(im_data, 0)
            im_data = (im_data - 127.5) * 0.0078125
            out = rnet(im_data)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            score = out1[1, :]
            ipass = score > threshold[1]
            ipass_x = torch.nonzero(ipass)[:, 0]
            total_boxes = torch.cat((total_boxes[ipass_x, 0:4], score[ipass].unsqueeze(1)), dim=1)
            mv = out0[:, ipass_x]
            if total_boxes.shape[0] > 0:
                pick = nms(total_boxes[:, :4], total_boxes[:, 4], 0.7)
                total_boxes = total_boxes[pick, :]
                total_boxes = bbreg(total_boxes, mv[:, pick].permute(1, 0))
                total_boxes = rerec(total_boxes)

        # Output network
        numbox = total_boxes.shape[0]
        if numbox > 0:
            total_boxes = total_boxes.int()
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes, w, h)
            im_data = []
            for k in range(0, numbox):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = img[:, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (48, 48)))
            im_data = torch.cat(im_data, 0)
            im_data = (im_data - 127.5) * 0.0078125
            out = onet(im_data)

            out0 = out[0].permute(1, 0)
            # out1 = out[1].permute(1, 0)
            out2 = out[2].permute(1, 0)
            score = out2[1, :]
            # points = out1
            ipass = score > threshold[2]
            ipass_x = torch.nonzero(ipass)[:, 0]
            # points = points[:, ipass_x]
            total_boxes = torch.cat((total_boxes[ipass_x, 0:4], score[ipass].unsqueeze(1)), dim=1)
            mv = out0[:, ipass_x]

            w_i = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h_i = total_boxes[:, 3] - total_boxes[:, 1] + 1
            # points[0:5, :] = w_i.repeat(5, 1) * points[0:5, :] + total_boxes[:, 0].repeat(5, 1) - 1
            # points[5:10, :] = h_i.repeat(5, 1) * points[5:10, :] + total_boxes[:, 1].repeat(5, 1) - 1
            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv.permute(1, 0))
                pick = nms(total_boxes[:, :4], total_boxes[:, 4], 0.7)
                total_boxes = total_boxes[pick, :]
                # points = points[:, pick]
        
        batch_boxes.append(total_boxes)

    return np.array(batch_boxes)
    

# def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
#     if not isinstance(imgs, Iterable):
#         imgs = [imgs]
#     if any(img.size != imgs[0].size for img in imgs):
#         raise Exception('MTCNN batch processing only compatible with equal-dimension images.')

#     imgs = [torch.as_tensor(np.uint8(img)).float().to(device) for img in imgs]
#     imgs = torch.stack(imgs).permute(0, 3, 1, 2)

#     batch_size = len(imgs)
#     h, w = imgs.shape[2:4]
#     m = 12.0 / minsize
#     minl = min(h, w)
#     minl = minl * m

#     # First stage
#     # Create scale pyramid
#     total_boxes_all = [np.empty((0, 9)) for i in range(batch_size)]
#     scale = m
#     while minl >= 12:
#         hs = int(h * scale + 1)
#         ws = int(w * scale + 1)
#         im_data = imresample(imgs, (hs, ws))
#         im_data = (im_data - 127.5) * 0.0078125
#         reg, probs = pnet(im_data)
        
#         for b_i in range(batch_size):
#             boxes = generateBoundingBox(reg[b_i], probs[b_i, 1], scale, threshold[0]).numpy()

#             # inter-scale nms
#             pick = nms(boxes, 0.5, 'Union')
#             if boxes.size > 0 and pick.size > 0:
#                 boxes = boxes[pick, :]
#                 total_boxes_all[b_i] = np.append(total_boxes_all[b_i], boxes, axis=0)

#         scale = scale * factor
#         minl = minl * factor

#     batch_boxes = []
#     for img, total_boxes in zip(imgs, total_boxes_all):
#         numbox = total_boxes.shape[0]
#         if numbox>0:
#             pick = nms(total_boxes, 0.7, 'Union')
#             total_boxes = total_boxes[pick, :]
#             regw = total_boxes[:, 2] - total_boxes[:, 0]
#             regh = total_boxes[:, 3] - total_boxes[:, 1]
#             qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
#             qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
#             qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
#             qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
#             total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
#             total_boxes = rerec(total_boxes)
#             total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
#             dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes, w, h)

#         numbox = total_boxes.shape[0]
#         if numbox>0:
#             # second stage
#             im_data = []
#             for k in range(0, numbox):
#                 if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
#                     img_k = img[:, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
#                     im_data.append(imresample(img_k, (24, 24)))
#             im_data = torch.cat(im_data, 0)
#             im_data = (im_data - 127.5) * 0.0078125
#             out = rnet(im_data)

#             out0 = np.transpose(out[0].numpy())
#             out1 = np.transpose(out[1].numpy())
#             score = out1[1, :]
#             ipass = np.where(score > threshold[1])
#             total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
#             mv = out0[:, ipass[0]]
#             if total_boxes.shape[0] > 0:
#                 pick = nms(total_boxes, 0.7, 'Union')
#                 total_boxes = total_boxes[pick, :]
#                 total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
#                 total_boxes = rerec(total_boxes.copy())

#         numbox = total_boxes.shape[0]
#         if numbox>0:
#             # third stage
#             total_boxes = np.fix(total_boxes).astype(np.int32)
#             dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
#             im_data = []
#             for k in range(0, numbox):
#                 if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
#                     img_k = img[:, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
#                     im_data.append(imresample(img_k, (48, 48)))
#             im_data = torch.cat(im_data, 0)
#             im_data = (im_data - 127.5) * 0.0078125
#             out = onet(im_data)

#             out0 = np.transpose(out[0].numpy())
#             out1 = np.transpose(out[1].numpy())
#             out2 = np.transpose(out[2].numpy())
#             score = out2[1, :]
#             points = out1
#             ipass = np.where(score>threshold[2])
#             points = points[:, ipass[0]]
#             total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
#             mv = out0[:,ipass[0]]

#             w_i = total_boxes[:, 2] - total_boxes[:, 0] + 1
#             h_i = total_boxes[:, 3] - total_boxes[:, 1] + 1
#             points[0:5, :] = np.tile(w_i, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
#             points[5:10, :] = np.tile(h_i, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
#             if total_boxes.shape[0] > 0:
#                 total_boxes = bbreg(total_boxes, np.transpose(mv))
#                 pick = nms(total_boxes, 0.7, 'Min')
#                 total_boxes = total_boxes[pick, :]
#                 points = points[:, pick]
        
#         batch_boxes.append(total_boxes)

#     return np.array(batch_boxes)


def bbreg(boundingbox,reg):
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = torch.stack((b1, b2, b3, b4)).permute(1, 0)

    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
    stride=2
    cellsize=12
    
    mask = probs >= thresh
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask.nonzero().float().flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    
    return boundingbox


def pad(boxes, w, h):
    device = 'cpu' if boxes.get_device() == -1 else boxes.get_device()

    tmpw = (boxes[:, 2] - boxes[:, 0] + 1)
    tmph = (boxes[:, 3] - boxes[:, 1] + 1)

    numbox = boxes.shape[0]
    dx = torch.ones(numbox, dtype=torch.int, device=device)
    dy = torch.ones(numbox, dtype=torch.int, device=device)
    edx = tmpw
    edy = tmph

    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    tmp = ex > w
    edx[tmp] = (-ex[tmp] + w + tmpw[tmp]).unsqueeze(1)
    ex[tmp] = w
    
    tmp = ey > h
    edy[tmp] = (-ey[tmp] + h + tmph[tmp]).unsqueeze(1)
    ey[tmp] = h

    tmp = x < 1
    dx[tmp] = (2 - x[tmp]).unsqueeze(1)
    x[tmp] = 1

    tmp = y < 1
    dy[tmp] = (2 - y[tmp]).unsqueeze(1)
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


def rerec(boxes):
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    l = torch.max(w, h)
    boxes[:, 0] = boxes[:, 0] + w * 0.5 - l * 0.5
    boxes[:, 1] = boxes[:, 1] + h * 0.5 - l * 0.5
    boxes[:, 2:4] = boxes[:, 0:2] + l.repeat(2, 1).permute(1, 0)

    return boxes


def imresample(img, sz):
    out_shape = (sz[0], sz[1])
    im_data = torch.nn.functional.interpolate(img, size=out_shape, mode='area')
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
        margin * (box[3] - box[1]) / (image_size - margin)
    ]
    box = [
        int(max(box[0] - margin[0]/2, 0)),
        int(max(box[1] - margin[1]/2, 0)),
        int(min(box[2] + margin[0]/2, img.size[0])),
        int(min(box[3] + margin[1]/2, img.size[1]))
    ]

    face = img.crop(box).resize((image_size, image_size), 2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path)+'/', exist_ok=True)
        save_args = {'compress_level': 0} if '.png' in save_path else {}
        face.save(save_path, **save_args)

    face = F.to_tensor(np.float32(face))
    
    return face
