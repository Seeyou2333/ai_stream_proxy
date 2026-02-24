import yaml
from fastapi import FastAPI

from src.api.ip_whitelist import IPWhitelistMiddleware
from src.api.routes import aistream
from src.mqtt.client import fast_mqtt
import src.mqtt.handlers  # 必须导入，触发装饰器

from src.manager.task_manager import InferenceTaskManager
from src.inference.gpu_manager import GpuManager
from config.logging_conf import setup_logger
from src.mqtt.handlers import register_mqtt_handlers

logger = setup_logger()

def create_app():
    app = FastAPI(title="视频流AI失败代理服务")
    app.add_middleware(IPWhitelistMiddleware)

    fast_mqtt.init_app(app)
    register_mqtt_handlers(app, fast_mqtt, logger)

    @app.on_event("startup")
    async def startup():
        with open("config/settings.yaml") as f:
            cfg = yaml.safe_load(f)
        app.state.logger=logger
        gpu_manager = GpuManager(cfg, logger)
        task_manager = InferenceTaskManager(cfg, gpu_manager, logger)

        aistream.init(task_manager)
        app.include_router(aistream.router)

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("服务关闭，停止所有任务")
        aistream.task_manager.stop_all()

    return app
