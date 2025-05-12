import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# 1) デバイス設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) MTCNNによる顔検出モデル
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# 3) Segformerによる顔パーシングモデル
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model     = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(DEVICE)  # skin class=1 :contentReference[oaicite:6]{index=6}

def get_face_parse_mask(frame: np.ndarray, bbox: list) -> np.ndarray:
    """顔バウンディングボックス内をSegformerでパースし、「肌」クラスのマスクを返す"""
    xmin, ymin, xmax, ymax = map(int, bbox)
    crop = frame[ymin:ymax, xmin:xmax]
    # PIL画像またはndarrayを直接渡せます
    inputs = processor(images=crop, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # (1, num_labels, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits, size=crop.shape[:2], mode="bilinear", align_corners=False
    )
    labels = upsampled.argmax(dim=1)[0].cpu().numpy()
    mask = (labels == 1)  # skin class index=1 :contentReference[oaicite:7]{index=7}
    full_mask = np.zeros(frame.shape[:2], dtype=bool)
    full_mask[ymin:ymax, xmin:xmax] = mask
    return full_mask

def enhance_face(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """HSV操作で露出＆血色向上＋バイラテラルフィルタで滑らか化"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v[mask] = np.clip(v[mask] * 1.2 + 15, 0, 255)    # +20% brightness + bias :contentReference[oaicite:8]{index=8}
    s[mask] = np.clip(s[mask] * 1.15, 0, 255)        # +15% saturation :contentReference[oaicite:9]{index=9}
    hsv2 = cv2.merge([h, s, v]).astype(np.uint8)
    bgr2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    # 肌滑らか化
    smooth = cv2.bilateralFilter(bgr2, d=9, sigmaColor=75, sigmaSpace=75)  # エッジ保持平滑化 :contentReference[oaicite:10]{index=10}
    out = frame.copy()
    out[mask] = smooth[mask]
    return out

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webカメラを開けません")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _  = mtcnn.detect(frame_rgb)
        output    = frame.copy()

        if boxes is not None:
            for bbox in boxes:
                mask = get_face_parse_mask(frame, bbox)
                output = enhance_face(output, mask)

        cv2.imshow("Actress Light with Complexion", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
