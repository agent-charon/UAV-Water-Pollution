import paho.mqtt.client as mqtt
import time
import json
import os
from datetime import datetime
from utils.logger import setup_logger
from utils.config_parser import ConfigParser
# This script would run on the Raspberry Pi integrated into the drone (as per paper Fig. 2)

# Responsibilities:
# 1. Act as a local MQTT broker (or connect to one if already running on RPi).
#    For simplicity, this script will assume an external broker is running (e.g., Mosquitto on RPi).
#    It will subscribe to NodeMCU sensor data.
# 2. Aggregate data from multiple NodeMCU nodes.
# 3. (As per paper) "processes the sensor readings and images of the water body,
#    transmitting them to the cloud for advanced analysis and secure storage".
#    This implies it might also get image paths or trigger image capture.
# 4. (As per paper) "The average hourly readings from sensors ... are sent to the R-Pi MQTT broker,
#    and the client transmits the data to a database using a 4G communication interface."
#    This script will simulate the aggregation and logging part. Cloud/DB upload is out of scope for this basic impl.

logger = setup_logger("rpi_gateway")
config = ConfigParser()

MQTT_BROKER_LOCAL = config.get("mqtt_broker", "localhost") # RPi itself or a broker it connects to
MQTT_PORT_LOCAL = config.get("mqtt_port", 1883)
NODEMCU_TOPIC_SUBSCRIBE = config.get("nodemcu_topic_base", "sensors/nodemcu/") + "+/+" # e.g. sensors/nodemcu/NodeMCU1/ph
RPI_AGGREGATED_TOPIC_BASE = config.get("rpi_sensor_topic_base", "sensors/rpi_gateway/") # For publishing aggregated data

# For "average hourly readings" - this is a simplified in-memory aggregation
sensor_data_buffer = {} # { (nodemcu_id, sensor_type): [ (timestamp, value), ... ], ... }
AGGREGATION_INTERVAL_SEC = 3600 # 1 hour as per paper, for testing use smaller
last_aggregation_time = time.time()

# Path for logging aggregated data (simulating DB transmission)
AGGREGATED_LOG_DIR = "data/processed/rpi_aggregated_logs"
if not os.path.exists(AGGREGATED_LOG_DIR):
    os.makedirs(AGGREGATED_LOG_DIR)

def on_connect_local(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"RPi Gateway connected to Local MQTT Broker: {MQTT_BROKER_LOCAL}")
        client.subscribe(NODEMCU_TOPIC_SUBSCRIBE)
        logger.info(f"RPi Gateway subscribed to: {NODEMCU_TOPIC_SUBSCRIBE}")
    else:
        logger.error(f"RPi Gateway failed to connect to local broker, return code {rc}")

def on_message_local(client, userdata, msg):
    timestamp = time.time()
    payload_str = msg.payload.decode()
    logger.debug(f"RPi Gateway received on {msg.topic}: {payload_str}")

    topic_parts = msg.topic.split('/')
    if len(topic_parts) >= 4 and topic_parts[0] == "sensors" and topic_parts[1] == "nodemcu":
        nodemcu_id = topic_parts[2]
        sensor_type = topic_parts[3]
        try:
            # Assuming plain numeric value for simplicity as per paper's pH/TDS
            value = float(payload_str)
            key = (nodemcu_id, sensor_type)
            if key not in sensor_data_buffer:
                sensor_data_buffer[key] = []
            sensor_data_buffer[key].append((timestamp, value))
            logger.info(f"Buffered: {key} -> {value}")
        except ValueError:
            logger.warning(f"Could not parse value '{payload_str}' from {msg.topic} as float.")
        except Exception as e:
            logger.error(f"Error processing message on RPi Gateway: {e}")

def aggregate_and_publish(client):
    global last_aggregation_time, sensor_data_buffer
    logger.info("Performing hourly aggregation...")
    current_time = time.time()
    aggregated_log_entry = {"timestamp": current_time, "iso_timestamp": datetime.now().isoformat(), "sensors": {}}

    for key, readings in sensor_data_buffer.items():
        nodemcu_id, sensor_type = key
        if readings:
            # Filter readings for the last aggregation interval (e.g., last hour)
            # For simplicity, let's average all current buffer, then clear
            avg_value = sum(r[1] for r in readings) / len(readings)
            logger.info(f"Aggregated for {key}: Avg = {avg_value:.2f} from {len(readings)} readings.")

            # Publish to an RPi-specific topic
            agg_topic = f"{RPI_AGGREGATED_TOPIC_BASE}{nodemcu_id}/{sensor_type}/avg"
            client.publish(agg_topic, payload=f"{avg_value:.2f}", qos=1)
            logger.info(f"Published aggregated data to {agg_topic}")

            # Store for logging
            if nodemcu_id not in aggregated_log_entry["sensors"]:
                aggregated_log_entry["sensors"][nodemcu_id] = {}
            aggregated_log_entry["sensors"][nodemcu_id][sensor_type] = {"avg": avg_value, "num_readings": len(readings)}

    # Log aggregated data
    if aggregated_log_entry["sensors"]:
        log_filename = os.path.join(AGGREGATED_LOG_DIR, f"rpi_hourly_avg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_filename, 'w') as f:
            json.dump(aggregated_log_entry, f, indent=2)
        logger.info(f"Logged aggregated data to {log_filename}")

    # Clear buffer for next interval
    sensor_data_buffer = {}
    last_aggregation_time = current_time


def run_gateway():
    client_local = mqtt.Client(client_id="rpi_gateway_module")
    client_local.on_connect = on_connect_local
    client_local.on_message = on_message_local

    try:
        logger.info(f"RPi Gateway attempting to connect to local MQTT broker at {MQTT_BROKER_LOCAL}:{MQTT_PORT_LOCAL}")
        client_local.connect(MQTT_BROKER_LOCAL, MQTT_PORT_LOCAL, 60)
        client_local.loop_start() # Start a background thread for network loop

        while True:
            current_time = time.time()
            if (current_time - last_aggregation_time) >= AGGREGATION_INTERVAL_SEC:
                aggregate_and_publish(client_local)
            time.sleep(10) # Check every 10 seconds

    except ConnectionRefusedError:
        logger.error(f"RPi Gateway: Connection refused to local broker. Is it running?")
    except KeyboardInterrupt:
        logger.info("RPi Gateway stopped by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in RPi Gateway: {e}")
    finally:
        client_local.loop_stop()
        client_local.disconnect()
        logger.info("RPi Gateway disconnected from local MQTT broker.")

if __name__ == "__main__":
    # Set a shorter aggregation interval for testing
    AGGREGATION_INTERVAL_SEC = 60 # Test with 1 minute
    logger.info(f"RPi Gateway starting with aggregation interval: {AGGREGATION_INTERVAL_SEC}s")
    run_gateway()