import cv2
import numpy as np
import onnxruntime as ort
import time
from datetime import datetime
#import spacemit_ort #SpaceMITExecutionProvider加速

class YOLOv12_ONNX_Inference:
    def __init__(self, onnx_path, input_size=(320, 320), conf_thres=0.5, iou_thres=0.45, score_thres=0.25):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.score_thres = score_thres

        # 初始化会话
        #providers = ['SpaceMITExecutionProvider', 'CPUExecutionProvider']
        providers = ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            print(f"SpaceMITExecutionProvider不可用，回退到CPU: {e}")
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_width, self.input_height = input_size

        # 你的类别列表
        self.classes = [
            "accipiter_nisus", "arenaria_interpres", "calidris_falcinellus", "calidris_tenuirostris",
            "calliope_calliope", "centropus_sinensis", "circus_spilonotus", "egetta_eulophotes",
            "egretta_sacra", "elanus_caeruleus", "falco_amurensis", "falco_tinnunculus",
            "garrulax_canorus", "halcyon_smyrnensis", "hydrophasianus_chirurgus", "leiothrix_argentauris",
            "leiothrix_lutea", "limnodromus_semipalmatus", "merops_philippinus", "milvus_migrans",
            "numenius_arquata", "pandion_haliaetus", "platalea_leucorodia", "platalea_minor"
        ]

    def preprocess(self, image):
        img_h, img_w, _ = image.shape
        scale = min(self.input_height / img_h, self.input_width / img_w)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_h, pad_w = (self.input_height - new_h) // 2, (self.input_width - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized_img
        input_tensor = (canvas[:, :, ::-1].transpose(2, 0, 1) / 255.0).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor, scale, pad_w, pad_h

    def postprocess(self, output, original_shape, scale, pad_w, pad_h):
        predictions = np.squeeze(output[0]).T

        # 简单适配 4+num_classes 格式
        if predictions.shape[1] == 4 + len(self.classes):
            class_scores = predictions[:, 4:]
        else:
            # 尝试适配 5+num_classes
            class_scores = predictions[:, 5:]

        max_class_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        keep = max_class_scores > self.conf_thres

        if not np.any(keep):
            return [], [], []

        box_preds = predictions[keep, :4]
        final_scores = max_class_scores[keep]
        final_class_ids = class_ids[keep]

        x1 = box_preds[:, 0] - box_preds[:, 2] / 2
        y1 = box_preds[:, 1] - box_preds[:, 3] / 2
        x2 = box_preds[:, 0] + box_preds[:, 2] / 2
        y2 = box_preds[:, 1] + box_preds[:, 3] / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        img_h, img_w = original_shape
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

        cv_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(cv_boxes, final_scores.tolist(), self.conf_thres, self.iou_thres)

        if len(indices) > 0:
            if isinstance(indices, np.ndarray): indices = indices.flatten()
            return boxes[indices].astype(int), final_scores[indices], final_class_ids[indices]
        return [], [], []

    def detect(self, image_path):
        """
        修改点：返回 (result_image, detected_label_string)
        """
        # 1. 记录推理开始时间
        Total_start_time = time.time()
        original_image = cv2.imread(image_path)
        if original_image is None: return None, None

        input_tensor, scale, pad_w, pad_h = self.preprocess(original_image)
        Inference_start_time = time.time()
        outputs = self.session.run([self.session.get_outputs()[0].name], {self.input_name: input_tensor})
        Inference_end_time = time.time()
        boxes, scores, class_ids = self.postprocess(outputs, original_image.shape[:2], scale, pad_w, pad_h)

        # 2. 记录推理结束时间并计算 FPS
        Total_end_time = time.time()
        Inference_time = (Inference_end_time - Inference_start_time) * 1000
        Total_time = (Total_end_time - Total_start_time) * 1000
        #fps = 1.0 / Inference_time if Inference_time > 0 else 0

        # 3. 获取当前时间字符串
        #current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 4. # 绘制时间 (左上角)
        #cv2.putText(original_image, f" {current_time_str}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

        # 绘制 FPS (紧接着时间下方)
        cv2.putText(original_image, f"Inference_time: {Inference_time:.0f}ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 5)
        cv2.putText(original_image, f"Total_time: {Total_time:.0f}ms", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # 绘制结果
        result_image = self.draw_results(original_image, boxes, scores, class_ids)

        # 获取最可信的 Label
        best_label = None
        if len(class_ids) > 0:
            # 简单策略：取第一个（通常 NMS 后分数最高的在前）
            best_id = class_ids[0]
            best_label = self.classes[best_id]

        return result_image, best_label

    def draw_results(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            label = f"{self.classes[class_id]}: {score:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=8)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness=6)
        return image