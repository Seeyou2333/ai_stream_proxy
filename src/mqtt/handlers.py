import json
from src.mqtt.client import fast_mqtt
from src.mqtt.topics import DEVICE_OSD

def register_mqtt_handlers(app, fast_mqtt, logger):

    @fast_mqtt.on_connect()
    def on_connect(client, flags, rc, properties):
        client.subscribe(DEVICE_OSD, qos=1)
        logger.info("MQTT 已连接，订阅设备 OSD")

    @fast_mqtt.on_message()
    async def on_message(client, topic, payload, qos, properties):
        try:
            data = json.loads(payload.decode())
            device_sn = topic.split("/")[2]

            # 示例逻辑
            # device = device_manager.get_or_create(device_sn)
            # device.update(data)

            logger.info(f"sn[{device_sn}] | data: {data}")

        except Exception:
            logger.exception("MQTT 消息处理失败")