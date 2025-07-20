r"""
    Receiver for PhoneIMU unity Android project.
"""

__all__ = ['WearableSensorSet']


import socket
import threading
from dataclasses import dataclass
import numpy as np


@dataclass
class Measurement:
    timestamp: float
    orientation: np.ndarray
    acceleration: np.ndarray
    raw_angular_velocity: np.ndarray
    raw_acceleration: np.ndarray
    raw_magnetic_field: np.ndarray
    compass_timestamp: float
    gps_position: np.ndarray
    gps_accuracy: np.ndarray
    gps_timestamp: float
    light_intensity: float
    proximity_distance: float
    pressure: float
    humidity: float
    ambient_temperature: float
    step_count: int

    def __repr__(self):
        return 'Measurement(t=%.3f)' % self.timestamp


class WearableSensorSet:
    g = 9.8

    def __init__(self, port=8989):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_address = ('0.0.0.0', port)
        sock.bind(server_address)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock = sock
        self.measurements = {}
        self.thread = threading.Thread(target=self._run)
        self.thread.setDaemon(True)
        self.thread.start()

    @staticmethod
    def _parse_signle_data(d: str):
        g = WearableSensorSet.g
        (sensor_id_, timestamp_, orientation_, acceleration_, raw_angular_velocity_, raw_acceleration_,
         raw_magnetic_field_, compass_timestamp_, gps_position_, gps_accuracy_, gps_timestamp_, light_intensity_,
         proximity_distance_, pressure_, humidity_, ambient_temperature_, step_count_) = d.split('#')
        sensor_id = int(sensor_id_)
        timestamp = float(timestamp_)
        orientation = [float(_) for _ in orientation_.split(' ')]                     # wxyz
        acceleration = [float(_) * g for _ in acceleration_.split(' ')]               # xyz
        raw_angular_velocity = [float(_) for _ in raw_angular_velocity_.split(' ')]   # xyz
        raw_acceleration = [float(_) * g for _ in raw_acceleration_.split(' ')]       # xyz
        raw_magnetic_field = [float(_) for _ in raw_magnetic_field_.split(' ')]       # xyz
        compass_timestamp = float(compass_timestamp_)
        gps_position = [float(_) for _ in gps_position_.split(' ')]    # lat-lon-alt
        gps_accuracy = [float(_) for _ in gps_accuracy_.split(' ')]    # hacc-vacc
        gps_timestamp = float(gps_timestamp_)
        light_intensity = float(light_intensity_)
        proximity_distance = float(proximity_distance_)
        pressure = float(pressure_)
        humidity = float(humidity_)
        ambient_temperature = float(ambient_temperature_)
        step_count = int(step_count_)
        return (sensor_id,
                Measurement(timestamp, np.array(orientation), np.array(acceleration), np.array(raw_angular_velocity),
                            np.array(raw_acceleration), np.array(raw_magnetic_field), compass_timestamp,
                            np.array(gps_position), np.array(gps_accuracy), gps_timestamp, light_intensity,
                            proximity_distance, pressure, humidity, ambient_temperature, step_count))

    def _run(self):
        data = ''
        while True:
            d, addr = self.sock.recvfrom(1024)
            data += d.decode()
            data_list = data.split('$')
            data = data_list[-1]
            for d in data_list[:-1]:
                sensor_id, sensor_measurement = self._parse_signle_data(d)
                self.measurements[sensor_id] = sensor_measurement

    def get(self):
        return self.measurements.copy()
