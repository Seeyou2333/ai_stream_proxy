import ffmpeg
import numpy as np
import threading
import queue
from src.core.ffmpeg_cfg import get_ffmpeg_args
import cv2
import datetime
import time
from src.utils.drawer import Drawer

class AiStreamer:
    def __init__(self, config, detector, logger,input_url,gpu_id):
        self.cfg = config
        self.detector = detector
        self.logger = logger
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = True
        self.input_url=input_url
        stream_suffix = input_url.split('/')[-1].split('?')[0]
        self.output_url = input_url.replace("rtsp://", "rtmp://") + "_ai"
        rtmp_base = config['video'].get('rtmp', 'rtmp://127.0.0.1:1935')
        base_url = rtmp_base.rstrip('/')
        self.output_url = f"{base_url}/live/{stream_suffix}_ai"
        self.w = config['video']['width']
        self.h = config['video']['height']
        self.frame_size = self.w * self.h * 3
        self.drawer = Drawer(config)
        self.gpu_id=gpu_id
        self.fps=0
        self.stream=f"{stream_suffix}_ai"
        self.last_heartbeat = time.time()

    def _reader(self):
        in_args, _ = get_ffmpeg_args(self.cfg,self.gpu_id)
        self.logger.info("启动 FFmpeg 解码器...")

        process_in = (
            ffmpeg
            .input(self.input_url, **in_args)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, cmd=self.cfg['paths']['ffmpeg_bin'])
        )
        try:
            while self.running:
                self.last_heartbeat = time.time()
                raw_frame = process_in.stdout.read(self.frame_size)
                if not raw_frame: 
                    self.logger.info("拉流结束退出")
                    break
                if self.frame_queue.full():
                    self.frame_queue.get() # 丢弃老帧
                self.frame_queue.put(raw_frame)
        finally:
            self.logger.info("正在关闭 FFmpeg 资源...")
            self.running = False
            process_in.stdout.close() 
            process_in.terminate()
            process_in.wait() 

    def _processor(self):
        _, out_args = get_ffmpeg_args(self.cfg, self.gpu_id)
        device_type = self.cfg.get('device_type', 'cpu').lower()
        process_out=None
        if(device_type!="ascend"):
            process_out = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{self.w}x{self.h}")
                .output(self.output_url, **out_args)
                .run_async(pipe_stdin=True, cmd=self.cfg['paths']['ffmpeg_bin'])
            )
        else:
            process_fixer = (
                ffmpeg
                .input('pipe:', format='h264') # 接收上一个 ffmpeg 的输出
                .output(self.output_url, vcodec='copy', format='flv', flvflags='no_duration_filesize',loglevel='error')
                .run_async(pipe_stdin=True, cmd=self.cfg['paths']['ffmpeg_bin'])
            )
            process_out = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{self.w}x{self.h}")
                .output('pipe:', **out_args) # 输出到 pipe，由 Python 中转给 process_fixer
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=self.cfg['paths']['ffmpeg_bin'])
            )

            def bridge_pipe(p_from, p_to):
                try:
                    while self.running:
                        data = p_from.stdout.read(4096) # 搬运块大小
                        if not data: break
                        p_to.stdin.write(data)
                except:
                    pass
                    
            bridge_thread = threading.Thread(target=bridge_pipe, args=(process_out, process_fixer), daemon=True)
            bridge_thread.start()
        


        frame_idx = 0
        metrics = {
            't_queue': 0.0, 't_infer': 0.0, 't_draw': 0.0, 
            't_write': 0.0, 't_total': 0.0, 'drop_count': 0
        }
        
        fps_target = 25
        frame_duration = 1.0 / fps_target
        
        window_start_time = time.perf_counter()
        last_push_time = time.perf_counter()
        
        detections, scale, pad_w, pad_h = [], 1.0, 0, 0

        while self.running:
            try:
                t_start = time.perf_counter() 

                # 队列排空 
                t0 = time.perf_counter()
                raw_bytes = None
                current_drop = 0
                # 取最后一帧
                while not self.frame_queue.empty():
                    raw_bytes = self.frame_queue.get_nowait()
                    current_drop += 1
                
                if raw_bytes is None:
                    try:
                        raw_bytes = self.frame_queue.get(timeout=2)
                    except queue.Empty:
                        continue
                
                t_queue_val = (time.perf_counter() - t0) * 1000 

                frame = np.frombuffer(raw_bytes, np.uint8).reshape([self.h, self.w, 3]).copy()

                # 推理
                t1 = time.perf_counter()
                t_infer_val = 0.0
                if frame_idx % 2 == 0:
                    detections, scale, pad_w, pad_h = self.detector.infer(frame)
                    t_infer_val = (time.perf_counter() - t1) * 1000
                
                # 绘图 
                t2 = time.perf_counter()
                frame = self.drawer.draw_detections(frame, detections, scale, pad_w, pad_h)
                t_draw_val = (time.perf_counter() - t2) * 1000

                elapsed = time.perf_counter() - last_push_time
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 推流
                t3 = time.perf_counter()
                process_out.stdin.write(frame.tobytes())
                last_push_time = time.perf_counter() 
                t_write_val = (time.perf_counter() - t3) * 1000

                # 统计总耗时
                t_total_val = (time.perf_counter() - t_start) * 1000

                metrics['t_queue'] += t_queue_val
                metrics['t_infer'] += t_infer_val
                metrics['t_draw'] += t_draw_val
                metrics['t_write'] += t_write_val
                metrics['t_total'] += t_total_val
                metrics['drop_count'] += current_drop

                # 每 25 帧打印一次平均表现
                if (frame_idx + 1) % 25 == 0:
                    now = time.perf_counter()
                    actual_window_duration = now - window_start_time
                    real_fps = 25.0 / actual_window_duration
                    
                    avg = {k: v / 25.0 for k, v in metrics.items()}
                    
                    self.fps=real_fps
                    # self.logger.info(
                    #     f"GPU[{self.gpu_id}] | "
                    #     f"RealFPS: {real_fps:4.1f} | "
                    #     f"处理总计: {avg['t_total']:4.1f}ms | "
                    #     f"单次推理: {avg['t_infer']*2:4.1f}ms | " 
                    #     f"绘图: {avg['t_draw']:4.1f}ms | "
                    #     f"推流: {avg['t_write']:4.1f}ms | "
                    #     f"AvgDrop: {avg['drop_count']:3.1f}f/loop"
                    # )
                    
                    # 重置
                    for k in metrics: metrics[k] = 0.0
                    window_start_time = now
                
                frame_idx += 1
                
            except Exception as e:
                self.logger.error(f"处理循环异常: {e}")
                if "Broken pipe" in str(e):
                    self.running = False
                continue

        # 退出后清理
        process_out.stdin.close()
        process_out.wait()

    def run(self):
        threads = [
            threading.Thread(target=self._reader, daemon=True),
            threading.Thread(target=self._processor, daemon=True)
        ]
        for t in threads: t.start()
        for t in threads: t.join()


    def get_video_info(self,url):
        try:
            probe = ffmpeg.probe(url)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            return width, height
        except Exception as e:
            print(f"获取视频尺寸失败: {e}")
            return None, None