from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frames = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Detect faces
    boxes, _ = mtcnn.detect(frames)
    
    # Display the results
    if boxes is not None:
        for top, right, bottom, left in boxes:
        # Draw a box around the face
            cv2.rectangle(frame, (int(top), int(left)), (int(bottom), int(right)), (0, 0, 255), 2)
    
    cv2.imshow('frame',frame)
    #print(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()