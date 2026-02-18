import time
from dataclasses import dataclass
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
    
@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    out_limit: float
    i_limit: float

    def __post_init__(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.prev_t = None

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.prev_t = None

    def update(self, e: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        
        # Integral
        self.i += e * dt
        if self.i > self.i_limit: self.i = self.i_limit
        if self.i < -self.i_limit: self.i = -self.i_limit

        # Derivative on error
        de = (e - self.prev_e) / dt
        self.prev_e = e

        u = self.kp * e + self.ki * self.i + self.kd * de

        # Output clamp
        if u > self.out_limit: u = self.out_limit
        if u < -self.out_limit: u = -self.out_limit
        return u
    
class LowPass:
    """1st-order low-pass: y = y + a*(x-y), a in (0..1)"""
    def __init__(self, alpha: float, init: float = 0.0):
        self.alpha = float(alpha)
        self.y = float(init)
        self.inited = False

    def reset(self, v: float = 0.0):
        self.y = float(v)
        self.inited = False

    def update(self, x: float) -> float:
        x = float(x)
        if not self.inited:
            self.y = x
            self.inited = True
            return self.y
        self.y = self.y + self.alpha * (x - self.y)
        return self.y


class SlewRateLimiter:
    """Limits rate of change: |dy/dt| <= rate"""
    def __init__(self, rate_per_s: float, init: float = 0.0):
        self.rate = float(rate_per_s)
        self.y = float(init)
        self.inited = False

    def reset(self, v: float = 0.0):
        self.y = float(v)
        self.inited = False

    def update(self, x: float, dt: float) -> float:
        x = float(x)
        if dt <= 0:
            return self.y
        if not self.inited:
            self.y = x
            self.inited = True
            return self.y

        max_step = self.rate * dt
        delta = x - self.y
        if delta > max_step: delta = max_step
        if delta < -max_step: delta = -max_step
        self.y += delta
        return self.y


def send_body_velocity(master, vx: float, vy: float, vz: float, yaw_rate: float):
    """
    BODY_NED frame:
      vx: forward (+)
      vy: right (+)
      vz: down (+)  -> for altitude hold, keep near 0
    yaw_rate: rad/s (+) is yaw right (generally)
    """
    # Ignore everything except vx,vy,vz and yaw_rate
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )

    master.mav.set_position_target_local_ned_send(
        0,  
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        type_mask,
        0, 0, 0,          # x,y,z position 
        vx, vy, vz,       # velocities
        0, 0, 0,          # accelerations 
        0,                # yaw 
        yaw_rate
    )
