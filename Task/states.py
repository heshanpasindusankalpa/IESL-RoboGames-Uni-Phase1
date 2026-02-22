from enum import Enum, auto

class DroneState(Enum):
    INITIALIZING = auto()
    ARMED = auto()
    TAKEOFF = auto()
    HOVER = auto()
    FOLLOW_LINE_01 = auto()

    APRILTAG_01_DETECTED = auto()
    SCAN_APRILTAG_01 = auto()
    TURN_RIGHT_90 = auto()

    FOLLOW_LINE_02 = auto()

    APRILTAG_02_DETECTED = auto()
    SCAN_APRILTAG_02 = auto()
    
    LANDING = auto()
    DONE = auto()