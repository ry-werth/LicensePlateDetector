import numpy as np
import cv2
import pytesseract as pt


INPUT_WIDTH = 640
INPUT_HEIGHT = 640

def get_detections(img,net):
    # 1.Turn the image insto a square using whatever is greater, the width or height
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. Resize and get predictions from model
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image,detections):
    
    # 3. Filter based on confidence
    
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    
    # 4.2 Non maximum suppresion (combine the overlapping boxes into one)
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)

    print(len(boxes_np), len(index))
    
    return boxes_np, confidences_np, index

# extrating text
def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return 'no number'
    
    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        
        return {"image": roi, "text": text}

def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    highest_conf = 0
    highest_conf_plate_text = ''
    highest_conf_plate_image = []
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text_dict = extract_text(image,boxes_np[ind])
        license_text = license_text_dict["text"]
        license_text_img = license_text_dict["image"]

        if bb_conf > highest_conf:
            highest_conf = bb_conf
            highest_conf_plate_text = license_text
            highest_conf_plate_image = license_text_img


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,153,51),2)
        cv2.rectangle(image,(x,y-40),(x+w,y),(255,153,51),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+40),(0,0,0),-1)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    return {"full_image":image, "plate_img": highest_conf_plate_image, "text": highest_conf_plate_text, "conf": highest_conf}

def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    final_dict = drawings(img,boxes_np,confidences_np,index)
    return final_dict