import torch
from torch import nn
import numpy as np
import os

from .utils.detect_face import detect_face, extract_face


class PNet(nn.Module):
    """MTCNN PNet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/pnet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """MTCNN RNet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/rnet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/onet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class MTCNN(nn.Module):
    """MTCNN face detection module.

    This class loads pretrained P-, R-, and O-nets and returns images cropped to include the face
    only, given raw input images of one of the following types:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) representing either a single image (3D) or a batch of images (4D).
    Cropped faces can optionally be saved to file
    also.
    
    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        min_face_size {int} -- Minimum face size to search for. (default: {20})
        thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
        factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
        post_process {bool} -- Whether or not to post process images tensors before returning.
            (default: {True})
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
            If False, the face with the highest detection probability is returned.
            (default: {True})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter. If a save_path is specified, the first face is saved to that
            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(
        self, image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        select_largest=True, keep_all=False, device=None
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, img, save_path=None, return_prob=False):
        """Run MTCNN face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces, returning tensors representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method below.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, or list.
        
        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved
                face image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether or not to return the detection probability.
                (default: {False})
        
        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra 
                dimension (batch) as the first dimension.

        Example:
        >>> from facenet_pytorch import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """

        # Detect faces
        with torch.no_grad():
            batch_boxes, batch_probs = self.detect(img)

        # Determine if a batch or single image was passed
        batch_mode = True
        if not isinstance(img, (list, tuple)) and not (isinstance(img, np.ndarray) and len(img.shape) == 4):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_probs = [batch_probs]
            batch_mode = False

        # Parse save path(s)
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]
        
        # Process all bounding boxes and probabilities
        faces, probs = [], []
        for im, box_im, prob_im, path_im in zip(img, batch_boxes, batch_probs, save_path):
            if box_im is None:
                faces.append(None)
                probs.append([None] if self.keep_all else None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                face = extract_face(im, box, self.image_size, self.margin, face_path)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]
                prob_im = prob_im[0]
            
            faces.append(faces_im)
            probs.append(prob_im)
    
        if not batch_mode:
            faces = faces[0]
            probs = probs[0]

        if return_prob:
            return faces, probs
        else:
            return faces

    def detect(self, img, landmarks=False):
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image or a list of PIL images.

        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})
        
        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.

        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes)
        probs = np.array(probs)
        points = np.array(points)

        if not isinstance(img, (list, tuple)) and not (isinstance(img, np.ndarray) and len(img.shape) == 4):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]

        if landmarks:
            return boxes, probs, points

        return boxes, probs


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y
