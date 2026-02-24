from fastapi_mqtt import FastMQTT, MQTTConfig

mqtt_config = MQTTConfig(
    host="192.168.1.253",
    port=11883,
    keepalive=60,
    username="",
    password=""
)

fast_mqtt = FastMQTT(config=mqtt_config)
