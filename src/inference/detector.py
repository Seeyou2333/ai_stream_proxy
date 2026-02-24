import cv2
import numpy as np
import onnxruntime as ort

class ONNXDetector:
    def __init__(self, config, gpu_id=0):
        self.gpu_id = gpu_id
        self.cfg = config['model']
        self.device_type = config.get('device_type', 'cpu').lower()

        providers = []
        
        if self.device_type == 'nvidia':
            providers.append((
                'CUDAExecutionProvider', {
                    'device_id': gpu_id,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024, 
                }
            ))
        elif self.device_type == 'ascend':
            providers.append((
                'CANNExecutionProvider', {
                    'device_id': gpu_id,
                    'arena_extend_strategy': 'kSameAsRequested',
                    "npu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "enable_cann_graph": True,
                }
            ))
        
        providers.append('CPUExecutionProvider')

        # self.session = ort.InferenceSession(self.cfg['path'], 
        #     providers=[
        #         ('CUDAExecutionProvider', {
        #             'device_id': gpu_id,
        #         }),
        #         'CPUExecutionProvider'
        #     ]
        # )

        try:
            self.session = ort.InferenceSession(self.cfg['path'], providers=providers)
            active_providers = self.session.get_providers()
            print(f"ONNX Session 启动成功。激活的 Providers: {active_providers}")
        except Exception as e:
            print(f"初始化 ONNX 失败，尝试回退到纯 CPU 模式。错误: {e}")
            self.session = ort.InferenceSession(self.cfg['path'], providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.cfg['input_size']

    def infer(self, frame):
        h, w = frame.shape[:2]
        # 1. 预处理 (Letterbox)
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_pad = (int(round(w * scale)), int(round(h * scale)))
        dw, dh = (self.input_size[1] - new_pad[0]) / 2, (self.input_size[0] - new_pad[1]) / 2
        
        img = cv2.resize(frame, new_pad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)), 
                                 int(round(dw - 0.1)), int(round(dw + 0.1)), 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        blob = img.transpose(2, 0, 1) # HWC to CHW
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
        
        # 2. 推理
        outputs = self.session.run(None, {self.input_name: blob})[0] # [1, 25200, 85]
        
        # 3. 后处理 (NMS)
        predictions = np.squeeze(outputs)
        # 过滤置信度 (第5列是 objectness)
        mask = predictions[:, 4] > self.cfg['conf_threshold']
        valid_hits = predictions[mask]
        
        boxes, confs, class_ids = [], [], []
        for hit in valid_hits:
            # 计算类分数: 类别独立分数 * 物体置信度
            class_scores = hit[5:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id] * hit[4]
            
            if score > self.cfg['conf_threshold']:
                cx, cy, bw, bh = hit[:4]
                # 转换为左上角坐标 [x, y, w, h]
                lx = cx - bw/2
                ly = cy - bh/2
                boxes.append([float(lx), float(ly), float(bw), float(bh)])
                confs.append(float(score))
                class_ids.append(int(class_id))

        # 执行非极大值抑制 (去重)
        indices = cv2.dnn.NMSBoxes(boxes, confs, self.cfg['conf_threshold'], self.cfg['iou_threshold'])
        
        final_dets = []
        if len(indices) > 0:
            for i in indices.flatten():
                b = boxes[i]
                # 封装为 6 个值的格式: [x1, y1, x2, y2, conf, cls_id]
                final_dets.append([b[0], b[1], b[0]+b[2], b[1]+b[3], confs[i], class_ids[i]])
        
        return final_dets, scale, dw, dh