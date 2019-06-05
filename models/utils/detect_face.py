import torch
import numpy as np
try:
    from cv2 import resize, INTER_AREA
    resize_args = {'interpolation': INTER_AREA}
except ImportError:
    from skimage.transform import resize
    resize_args = {'preserve_range': True, 'anti_aliasing': True}


def detect_face(img, minsize, pnet, rnet, onet, threshold, factor, device):
    w, h = img.size
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m
    img = np.uint8(img)

    # First stage
    # Create scale pyramid
    total_boxes = np.empty((0, 9))
    scale = m
    while minl >= 12:
        hs = int(h * scale + 1)
        ws = int(w * scale + 1)
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0, 3, 1, 2))
        reg, probs = pnet(torch.tensor(img_y).float().to(device))
        
        boxes = generateBoundingBox(reg[0], probs[0, 1], scale, threshold[0]).numpy()

        scale = scale * factor
        minl = minl * factor
        
        # inter-scale nms
        pick = nms(boxes, 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox>0:
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick,:]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox>0:
        # second stage
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[(dy[k] - 1):edy[k], (dx[k] - 1):edx[k], :] = img[(y[k] - 1):ey[k], (x[k] - 1):ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 2, 0, 1))
        out = rnet(torch.tensor(tempimg1).float().to(device))
        out0 = np.transpose(out[0].numpy())
        out1 = np.transpose(out[1].numpy())
        score = out1[1, :]
        ipass = np.where(score>threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox>0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[(dy[k] - 1):edy[k], (dx[k] - 1):edx[k], :] = img[(y[k] - 1):ey[k], (x[k] - 1):ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 2, 0, 1))
        out = onet(torch.tensor(tempimg1).float().to(device))
        out0 = np.transpose(out[0].numpy())
        out1 = np.transpose(out[1].numpy())
        out2 = np.transpose(out[2].numpy())
        score = out2[1, :]
        points = out1
        ipass = np.where(score>threshold[2])
        points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:,ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]
                    
    return total_boxes


def bbreg(boundingbox,reg):
    if reg.shape[1]==1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:,2]-boundingbox[:,0]+1
    h = boundingbox[:,3]-boundingbox[:,1]+1
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
    stride=2
    cellsize=12
    
    mask = probs >= thresh
    score = probs[mask]
    reg = np.transpose(reg[0:4, mask])
    bb = mask.nonzero().float().flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox
 

def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick


def pad(total_boxes, w, h):
    tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)
    tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:,0].copy().astype(np.int32)
    y = total_boxes[:,1].copy().astype(np.int32)
    ex = total_boxes[:,2].copy().astype(np.int32)
    ey = total_boxes[:,3].copy().astype(np.int32)

    tmp = np.where(ex>w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
    ex[tmp] = w
    
    tmp = np.where(ey>h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
    ey[tmp] = h

    tmp = np.where(x<1)
    dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
    x[tmp] = 1

    tmp = np.where(y<1)
    dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


def rerec(bboxA):
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
    l = np.maximum(w, h)
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
    return bboxA


def imresample(img, sz):
    im_data = resize(img, (sz[1], sz[0]), **resize_args)
    return im_data
