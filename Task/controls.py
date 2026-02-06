from pymavlink import mavutil

def force_arm(master): 
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,      # 1 to arm, 0 to disarm
        21196,  # Force-arm magic number for ArduPilot
        0, 0, 0, 0, 0
    )

def is_armed(master) -> bool:
    hb = master.recv_match(type='HEARTBEAT', blocking=False)
    if not hb:
        return False
    return (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

def takeoff(master, target_alt):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,
        0, 0,
        target_alt
    )

class AltitudeTracker:
    def __init__(self):
        self.stable_count = 0
        self.last_alt = None

    def update(self, master, target_alt, tol):
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if not msg:
            return False, self.last_alt
        
        self.last_alt = msg.relative_alt / 1000.0
        if self.last_alt >= (target_alt - tol):
            self.stable_count += 1
        else:
            self.stable_count = 0

        return self.stable_count >= 10, self.last_alt