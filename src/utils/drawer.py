import cv2
import numpy as np

class Drawer:
    def __init__(self, cfg):
        # 标签列表，建议使用英文标签或 ID，以匹配 cv2.putText
        self.label_list = cfg.get('labels', []) 
        draw_cfg = cfg.get('draw_config', {})
        
        # 直接定义 BGR 颜色 (例如绿色: (0, 255, 0))
        color_list = draw_cfg.get('color', [0, 255, 0])
        self.color_bgr = tuple(color_list)
        
        self.thickness = 2
        # OpenCV 字体缩放比例
        self.font_scale = 0.6 

    def draw_detections(self, frame, detections, scale, pad_w, pad_h):
        """
        frame: numpy ndarray (BGR 格式)
        detections: 推理结果 [x1, y1, x2, y2, conf, cls_id]
        """
        if len(detections) == 0:
            return frame

        h_img, w_img = frame.shape[:2]

        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls_id = det
                
                # --- 1. 坐标还原 ---
                orig_x1 = max(0, int((x1 - pad_w) / scale))
                orig_y1 = max(0, int((y1 - pad_h) / scale))
                orig_x2 = min(w_img, int((x2 - pad_w) / scale))
                orig_y2 = min(h_img, int((y2 - pad_h) / scale))
                
                # --- 2. 绘制矩形框 (原地修改内存) ---
                cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), 
                              self.color_bgr, self.thickness)

                # --- 3. 绘制英文标签与置信度 ---
                idx = int(cls_id)
                name = self.label_list[idx] if idx < len(self.label_list) else f"ID:{idx}"
                label_text = f"{name} {conf:.2f}"
                
                # 文本背景（可选，增加可读性）
                # (t_w, t_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
                # cv2.rectangle(frame, (orig_x1, orig_y1 - t_h - 5), (orig_x1 + t_w, orig_y1), self.color_bgr, -1)
                
                # 直接绘制文字
                cv2.putText(frame, label_text, (orig_x1, max(0, orig_y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1)

        return frame