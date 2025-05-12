import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN # 顔検出用
from PIL import Image
import torchvision.transforms as T

# 0. 設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 女優ライト効果のパラメータ
BRIGHTNESS_BASE_INCREASE_FACTOR = 1.10
BRIGHTNESS_CENTER_BOOST_FACTOR = 1.05
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
MASK_BLUR_KERNEL_SIZE = (31, 31) # 解像度が上がったので少しぼかしを強くしても良いかも
FACE_DETECTION_CONFIDENCE = 0.95

# 1. モデルのロード
mtcnn = MTCNN(keep_all=True, device=DEVICE, select_largest=False, thresholds=[0.6, 0.7, 0.7], factor=0.709, min_face_size=20)
print("MTCNN model loaded.")

# 2. 「女優ライト」効果を適用する関数 (前回と同じ)
def apply_actress_light_effect(image_bgr, face_bbox):
    output_image = image_bgr.copy()
    (xmin, ymin, xmax, ymax) = [int(b) for b in face_bbox]
    original_height, original_width = image_bgr.shape[:2]

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(original_width, xmax)
    ymax = min(original_height, ymax)

    face_roi_h = ymax - ymin
    face_roi_w = xmax - xmin

    if face_roi_h <= 0 or face_roi_w <= 0:
        return image_bgr

    face_mask_canvas = np.zeros((original_height, original_width), dtype=np.float32)
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    axis_x = face_roi_w // 2
    axis_y = face_roi_h // 2
    
    if axis_x > 0 and axis_y > 0:
        cv2.ellipse(face_mask_canvas, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)

    blurred_face_mask = cv2.GaussianBlur(face_mask_canvas, MASK_BLUR_KERNEL_SIZE, 0)
    max_val = np.max(blurred_face_mask)
    if max_val > 0:
        blurred_face_mask = blurred_face_mask / max_val
    
    processed_bgr_image = image_bgr.copy()

    hsv_image_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image_full)
    v_channel_float = v_channel.astype(np.float32)

    v_face_roi = v_channel_float[ymin:ymax, xmin:xmax]

    if v_face_roi.size > 0:
        y_coords, x_coords = np.ogrid[:face_roi_h, :face_roi_w]
        center_y_roi, center_x_roi = face_roi_h // 2, face_roi_w // 2
        
        sigma_gauss = min(face_roi_h, face_roi_w) / 3.5 
        if sigma_gauss < 1.0: sigma_gauss = 1.0
        
        variance = sigma_gauss**2
        gauss_weights = np.exp(-(((y_coords - center_y_roi)**2 + (x_coords - center_x_roi)**2) / (2. * variance)))
        
        brightness_factor_map_roi = BRIGHTNESS_BASE_INCREASE_FACTOR * \
                                    (1.0 + (BRIGHTNESS_CENTER_BOOST_FACTOR - 1.0) * gauss_weights)
        
        v_adjusted_roi = np.clip(v_face_roi * brightness_factor_map_roi, 0, 255)
        
        v_temp_full = v_channel_float.copy()
        v_temp_full[ymin:ymax, xmin:xmax] = v_adjusted_roi
        
        adjusted_hsv_image_full = cv2.merge([h_channel, s_channel, v_temp_full.astype(np.uint8)])
        processed_bgr_image = cv2.cvtColor(adjusted_hsv_image_full, cv2.COLOR_HSV2BGR)

    face_roi_for_smoothing = processed_bgr_image[ymin:ymax, xmin:xmax]
    if face_roi_for_smoothing.size > 0:
        smoothed_face_roi = cv2.bilateralFilter(face_roi_for_smoothing, 
                                                BILATERAL_D, 
                                                BILATERAL_SIGMA_COLOR, 
                                                BILATERAL_SIGMA_SPACE)
        processed_bgr_image[ymin:ymax, xmin:xmax] = smoothed_face_roi
    
    alpha_mask_3ch = cv2.cvtColor(blurred_face_mask, cv2.COLOR_GRAY2BGR)
    output_image_float = image_bgr.astype(np.float32) * (1.0 - alpha_mask_3ch) + \
                         processed_bgr_image.astype(np.float32) * alpha_mask_3ch
    output_image = np.clip(output_image_float, 0, 255).astype(np.uint8)
    
    return output_image

# 3. メイン処理 (ウェブカメラ用) - 解像度設定を追加
def main_camera():
    cap = cv2.VideoCapture(0) # 0はデフォルトのカメラ
    if not cap.isOpened():
        print("エラー: カメラが見つからないか、開けませんでした。")
        return

    # --- 解像度設定 (ここから変更/追加) ---
    # 希望する解像度 (例: 1280x720 HD)
    # お使いのカメラがサポートする解像度を指定してください。
    # 一般的な解像度: 640x480 (VGA), 1280x720 (HD), 1920x1080 (Full HD)
    desired_width = 1280
    desired_height = 720

    # カメラに解像度を設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # 実際に設定された解像度を取得して確認
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"希望解像度: {desired_width}x{desired_height}, 実際に設定された解像度: {actual_width}x{actual_height}")

    if actual_width != desired_width or actual_height != desired_height:
        print("注意: カメラが希望の解像度をサポートしていないか、他の要因で異なる解像度が設定されました。")
        print("サポートされている解像度で動作します。")
    # --- 解像度設定 (ここまで変更/追加) ---

    print("「大物女優カメラ」起動！ 'q'キーで終了します。")
    cv2.namedWindow('Oomono Joyu Camera', cv2.WINDOW_NORMAL)
    # ウィンドウの初期サイズを実際のフレームサイズに合わせる（任意）
    if actual_width > 0 and actual_height > 0:
         cv2.resizeWindow('Oomono Joyu Camera', actual_width, actual_height)


    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("エラー: フレームを取得できませんでした。")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
        
        processed_frame = frame_bgr.copy()

        if boxes is not None:
            for i, bbox in enumerate(boxes):
                if probs[i] is not None and probs[i] > FACE_DETECTION_CONFIDENCE:
                    processed_frame = apply_actress_light_effect(processed_frame, bbox)
        
        cv2.imshow('Oomono Joyu Camera', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("カメラを終了しました。")

# 画像ファイル処理用 (前回と同じ)
def main_image(image_path="sample_face.jpg"):
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        print(f"エラー: 画像が見つかりません: {image_path}")
        try:
            import skimage.data
            img_astronaut = skimage.data.astronaut()
            face_portion = img_astronaut[50:250, 150:300]
            frame_bgr = cv2.cvtColor(face_portion, cv2.COLOR_RGB2BGR)
            cv2.imwrite("sample_face.jpg", frame_bgr)
            print(f"サンプル画像 'sample_face.jpg' を作成しました。再度実行してください。")
        except ImportError:
            print("skimage がないのでサンプル画像を自動生成できませんでした。手動で 'sample_face.jpg' を用意してください。")
        except Exception as e:
            print(f"サンプル画像作成中にエラー: {e}")
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
    
    processed_frame = frame_bgr.copy()

    if boxes is not None:
        for i, bbox in enumerate(boxes):
            if probs[i] is not None and probs[i] > FACE_DETECTION_CONFIDENCE:
                processed_frame = apply_actress_light_effect(processed_frame, bbox)
    
    cv2.imshow('Original Image', frame_bgr)
    cv2.imshow('Oomono Joyu Camera - Image', processed_frame)
    print("画像処理完了。いずれかのキーを押してウィンドウを閉じてください。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mode = "camera" # "camera" または "image"
    
    if mode == "camera":
        main_camera()
    elif mode == "image":
        main_image("sample_face.jpg") 
    else:
        print("無効なモードが選択されました。")
