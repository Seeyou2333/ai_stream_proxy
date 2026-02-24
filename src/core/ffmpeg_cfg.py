#cpu环境
def _get_cpu_config(cfg):
    input_args = {
        'rtsp_transport': 'tcp',
        'threads': 'auto',          
        'probesize': '32',
        'analyzeduration': '0',
        'fflags': 'nobuffer+discardcorrupt',
        'flags': 'low_delay',
        'loglevel': 'error',
    }

    output_args = {
        'format': 'rtsp',
        'vcodec': 'libx264',
        'pix_fmt': 'yuv420p',        
        'preset': 'ultrafast',  
        'tune': 'zerolatency',  
        'profile:v': 'baseline',
        'level': '4.1',         
        'g': '50',              
        'x264-params': 'repeat-headers=1:annexb=1',
        'b:v': cfg.get('bitrate', '2M'),
        'maxrate': cfg.get('bitrate', '2M'),
        'bufsize': '4M',
        'rtsp_transport': 'tcp',
        'loglevel': 'error',
    }
    return input_args, output_args

#nvidia 环境
def _get_nvidia_config(cfg, gpu_id=0):

    target_res = f"{cfg['video']['width']}x{cfg['video']['height']}"
    
    input_args = {
        'rtsp_transport': 'tcp',
        'hwaccel': 'cuda',
        'c:v': 'h264_cuvid',
        'hwaccel_device': str(gpu_id),
        # 'probesize': '32',          
        # 'analyzeduration': '0',     
        'fflags': 'nobuffer', 
        #'flags': 'low_delay',
        'resize': target_res,

         'loglevel': 'error',
    }
    
    output_args = {
        'format': 'flv',
        'vcodec': 'h264_nvenc',
        'gpu': str(gpu_id),
        'pix_fmt': 'yuv420p',
        'profile:v': 'baseline',
        'level': '4.1',
        'bf': '0',
        'g': '50',               
        'forced-idr': '1',      
        'bsf:v': 'dump_extra',
        #'bsf:v': 'h264_mp4toannexb,dump_extra',  #h264_nvenc避免webrtc拉流无法解码
        'flags': '+global_header', #h264_nvenc避免webrtc拉流无法解码
        'preset': 'p4',
        'tune': 'ull',
        'delay': '0',
        'rc': 'cbr',
        'b:v': "4M",
        'maxrate': "5M",
        'bufsize': '2M',
        
        'loglevel': 'error',
    }
    return input_args, output_args

#ascend 环境
def _get_ascend_config(cfg,gpu_id=0):

    target_res = f"{cfg['video']['width']}x{cfg['video']['height']}"

    input_args = {
        'rtsp_transport': 'tcp',
        'hwaccel': 'ascend', 
        'c:v': 'h264_ascend',
        'device_id': str(gpu_id),
        'channel_id': '0',
        #'probesize': '32',
        #'analyzeduration': '0',
        'fflags': 'nobuffer',
        #'flags': 'low_delay',
        'resize': target_res,
        #'loglevel': 'error',
        #'stimeout': '5000000',        # RTSP 超时设置 (5秒，单位微秒)
        #'reconnect': '1',             # 开启重连
        #'reconnect_at_eof': '1',
        #'reconnect_streamed': '1',
        #'reconnect_delay_max': '2',
    }
    
    output_args = {
        # 'format': 'flv',
        # 'vcodec': 'h264_ascend', 
        # 'device_id': str(gpu_id),
        # 'channel_id': '1',
        # 'profile': '0',
        # 'rc_mode': '0',
        # 'g': '50',
        # #'frame_rate': '25',
        # 'max_bit_rate': '2000',
        # 'movement_scene': '1',
        # #'bsf:v': 'h264_mp4toannexb,dump_extra',
        # 'flags': '+global_header',
        'loglevel': 'error',
        'vcodec': 'h264_ascend', 
        'device_id': str(gpu_id),
        'channel_id': '1',
        'profile': '0',   # Baseline
        'rc_mode': '0',
        'g': '50',
        'max_bit_rate': cfg.get('bitrate', '2000'),
        'movement_scene': '1',
        'format': 'h264', # 重要：输出裸流格式进行中转
    }
    return input_args, output_args



def get_ffmpeg_args(cfg, gpu_id=0):
    device_type = cfg.get('device_type', 'cpu').lower()
    
    if device_type == 'nvidia':
        return _get_nvidia_config(cfg,gpu_id)
    elif device_type == 'ascend':
        return _get_ascend_config(cfg,gpu_id)
    else:
        return _get_cpu_config(cfg)