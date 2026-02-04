from enum import Enum, auto

class DroneState(Enum):
    INITIALIZING = auto()
    ARMED = auto()
    TAKEOFF = auto()
    HOVER = auto()
    FOLLOW_LINE_01 = auto()
    LANDING = auto()
    DONE = auto()