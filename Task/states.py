from enum import Enum, auto

class DroneState(Enum):
    INITIALIZING = auto()
    ARMED = auto()
    TAKEOFF = auto()
    HOVER = auto()
    LANDING = auto()
    DONE = auto()