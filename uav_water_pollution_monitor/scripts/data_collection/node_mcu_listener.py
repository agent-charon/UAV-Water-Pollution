import paho.mqtt.client as mqtt
import time
import json
import os
from datetime import datetime
from utils.logger import setup_logger
from utils.config_parser import ConfigParser

# This script runs on a machine that can receive MQTT messages from NodeMCUs.
# It's a basic listener that logs the received data.
# The RPi_gateway.py will be a more sophisticated broker/forwarder.

logger = setup_logger("nodemcu_listener")
config = ConfigParser()

MQTT_BROKER = config.get("mqtt_broker", "localhost")
MQTT_PORT = config.get("mqtt_port", 1883)
NODEMCU_TOPIC_BASE = config.get("nodemcu_topic_base", "sensors/nodemcu/") # e.g., sensors/nodemcu/+/+
LOG_DIR = "data/raw/sensor_logs"
LOG_FILENAME_PREFIX = "nodemcu_sensor_data"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Current log file, will rotate daily or on script restart
current_log_file = None
current_log_date = None

def get_log_file():
    global current_log_file, current_log_date
    today_str = datetime.now().strftime("%Y%m%d")
    if current_log_date != today_str or current_log_file is None:
        current_log_date = today_str
        filename = os.path.join(LOG_DIR, f"{LOG_FILENAME_PREFIX}_{today_str}.csv")
        current_log_file = filename
        if not os.path.exists(current_log_file):
            with open(current_log_file, 'w') as f:
                f.write("timestamp,iso_timestamp,nodemcu_id,sensor_type,value,raw_payload\n")
        logger.info(f"Logging NodeMCU data to: {current_log_file}")
    return current_log_file

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"Connected to MQTT Broker: {MQTT_BROKER}")
        # Subscribe to all topics under the NodeMCU base, e.g., sensors/nodemcu/node1/ph, sensors/nodemcu/node2/tds
        # The paper mentions NodeMCU1 for pH and NodeMCU2 for TDS
        # A more general subscription:
        subscribe_topic = f"{NODEMCU_TOPIC_BASE}+/+" # + is a single-level wildcard
        client.subscribe(subscribe_topic)
        logger.info(f"Subscribed to topic: {subscribe_topic}")
    else:
        logger.error(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    timestamp = time.time()
    iso_timestamp = datetime.now().isoformat()
    payload_str = msg.payload.decode()
    logger.debug(f"Received message: {msg.topic} -> {payload_str}")

    # Expected topic structure: sensors/nodemcu/<nodemcu_id>/<sensor_type>
    # Example: sensors/nodemcu/NodeMCU1/ph
    topic_parts = msg.topic.split('/')
    nodemcu_id = "unknown_node"
    sensor_type = "unknown_sensor"

    if len(topic_parts) >= 4 and topic_parts[0] == "sensors" and topic_parts[1] == "nodemcu":
        nodemcu_id = topic_parts[2]
        sensor_type = topic_parts[3]

    try:
        # Assuming payload is a simple value or JSON
        try:
            data = json.loads(payload_str)
            # If JSON, extract relevant value if needed, or log the whole JSON
            # For simplicity, let's assume the paper's NodeMCUs send plain values for pH/TDS
            value = payload_str # Or data['value'] if it's a JSON like {"value": 7.5}
        except json.JSONDecodeError:
            value = payload_str # Treat as plain text/number

        log_file_path = get_log_file()
        with open(log_file_path, 'a') as f:
            f.write(f"{timestamp},{iso_timestamp},{nodemcu_id},{sensor_type},{value},{payload_str}\n")

    except Exception as e:
        logger.error(f"Error processing message from {msg.topic}: {e}")

def run_listener():
    client = mqtt.Client(client_id="nodemcu_data_logger")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        logger.info(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Ensure MQTT broker is running at {MQTT_BROKER}:{MQTT_PORT}.")
    except KeyboardInterrupt:
        logger.info("Listener stopped by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        client.disconnect()
        logger.info("Disconnected from MQTT broker.")

if __name__ == "__main__":
    run_listener()