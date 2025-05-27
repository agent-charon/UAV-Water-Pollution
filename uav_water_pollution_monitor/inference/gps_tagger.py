import exif # Python library for reading and writing EXIF data
import os
from datetime import datetime
from utils.logger import setup_logger

# This script is for tagging images with GPS information.
# The actual GPS data would come from the UAV's flight controller.
# For this simulation, we might use placeholder GPS data or read from a log.

logger = setup_logger("gps_tagger")

def decimal_to_dms(decimal_degrees):
    """Converts decimal degrees to degrees, minutes, seconds tuple."""
    degrees = int(decimal_degrees)
    minutes_float = (decimal_degrees - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return (degrees, minutes, seconds)

def tag_image_with_gps(image_path, latitude, longitude, altitude=None, timestamp=None):
    """
    Adds GPS EXIF tags to an image file.
    Overwrites the image file with the new EXIF data.

    Args:
        image_path (str): Path to the JPEG image file.
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        altitude (float, optional): Altitude in meters.
        timestamp (datetime, optional): Timestamp of the capture.

    Returns:
        bool: True if tagging was successful, False otherwise.
    """
    if not os.path.exists(image_path) or not image_path.lower().endswith(('.jpg', '.jpeg')):
        logger.error(f"Invalid image path or not a JPEG: {image_path}")
        return False

    try:
        with open(image_path, 'rb') as img_file:
            img_exif = exif.Image(img_file)

        # Latitude
        lat_dms = decimal_to_dms(abs(latitude))
        img_exif.gps_latitude = lat_dms
        img_exif.gps_latitude_ref = "N" if latitude >= 0 else "S"

        # Longitude
        lon_dms = decimal_to_dms(abs(longitude))
        img_exif.gps_longitude = lon_dms
        img_exif.gps_longitude_ref = "E" if longitude >= 0 else "W"

        # Altitude
        if altitude is not None:
            img_exif.gps_altitude = altitude
            img_exif.gps_altitude_ref = exif.GpsAltitudeRef.ABOVE_SEA_LEVEL # Or 1 for below

        # Timestamp (UTC)
        if timestamp is None:
            timestamp = datetime.utcnow() # Use current UTC time if not provided
        
        # EXIF GPS timestamp is (hours, minutes, seconds)
        img_exif.gps_timestamp = (timestamp.hour, timestamp.minute, timestamp.second)
        img_exif.gps_datestamp = timestamp.strftime("%Y:%m:%d") # Standard EXIF date format

        # Overwrite the image with new EXIF data
        with open(image_path, 'wb') as new_img_file:
            new_img_file.write(img_exif.get_file())
        
        logger.info(f"Successfully tagged {os.path.basename(image_path)} with GPS: Lat={latitude:.6f}, Lon={longitude:.6f}")
        return True

    except AttributeError as ae:
        logger.error(f"AttributeError processing EXIF for {image_path}. Image might not have existing EXIF or is corrupted: {ae}")
        # This can happen if img_exif doesn't have gps attributes initially.
        # The `exif` library should create them if they don't exist when you assign.
        # Let's re-check if the library handles this gracefully or if we need to ensure attributes exist.
        # The library is designed to add these if they don't exist. The error might be elsewhere.
        return False
    except Exception as e:
        logger.error(f"Error tagging image {image_path} with GPS: {e}")
        return False

def read_gps_tags(image_path):
    """Reads GPS EXIF tags from an image."""
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
    try:
        with open(image_path, 'rb') as img_file:
            img_exif = exif.Image(img_file)
        
        if img_exif.has_exif and hasattr(img_exif, "gps_latitude"):
            gps_info = {
                "latitude": img_exif.gps_latitude,
                "latitude_ref": img_exif.gps_latitude_ref,
                "longitude": img_exif.gps_longitude,
                "longitude_ref": img_exif.gps_longitude_ref,
            }
            if hasattr(img_exif, "gps_altitude"):
                gps_info["altitude"] = img_exif.gps_altitude
                gps_info["altitude_ref"] = img_exif.gps_altitude_ref
            if hasattr(img_exif, "gps_timestamp"):
                 gps_info["timestamp"] = img_exif.gps_timestamp
            if hasattr(img_exif, "gps_datestamp"):
                 gps_info["datestamp"] = img_exif.gps_datestamp
            return gps_info
        else:
            logger.info(f"No GPS EXIF data found in {image_path}.")
            return None
    except Exception as e:
        logger.error(f"Error reading GPS tags from {image_path}: {e}")
        return None


if __name__ == '__main__':
    # Create a dummy JPEG image for testing
    dummy_jpeg_path = "temp_gps_test_image.jpg"
    if not os.path.exists(dummy_jpeg_path):
        try:
            from PIL import Image as PILImage
            img_pil = PILImage.new('RGB', (100, 100), color = 'cyan')
            img_pil.save(dummy_jpeg_path, "JPEG") # Must be JPEG for EXIF
            logger.info(f"Created dummy JPEG: {dummy_jpeg_path}")
        except ImportError:
            logger.error("Pillow not installed, cannot create dummy JPEG for GPS tagging test.")
            dummy_jpeg_path = None
        except Exception as e:
            logger.error(f"Error creating dummy JPEG: {e}")
            dummy_jpeg_path = None

    if dummy_jpeg_path and os.path.exists(dummy_jpeg_path):
        # Example GPS data (San Francisco)
        lat_test = 37.7749
        lon_test = -122.4194
        alt_test = 10.5 # meters
        
        logger.info(f"Attempting to tag image: {dummy_jpeg_path}")
        success = tag_image_with_gps(dummy_jpeg_path, lat_test, lon_test, alt_test)

        if success:
            logger.info("Reading back GPS tags...")
            read_tags = read_gps_tags(dummy_jpeg_path)
            if read_tags:
                logger.info(f"Read GPS Info: {read_tags}")
            else:
                logger.error("Failed to read back GPS tags.")
        else:
            logger.error("Failed to tag image with GPS.")
        
        # Clean up
        # os.remove(dummy_jpeg_path)
    else:
        logger.warning("Dummy JPEG for GPS tagging test was not created or found.")