# drone_dynamics_two_panels.py
# ---------------------------------------------------------------
# A test script for a 2D drone dynamic model with a two-panel display:
#  - Left panel: a handle the user moves with mouse or haptic device
#  - Right panel: a drone dot that responds to the handle's input

import sys
import math
import time
import numpy as np
import pygame

###############################################################################
# Minimal PHYSICS class (similar to PA1)
###############################################################################
try:
    import serial.tools.list_ports
    REAL_DEVICE_SUPPORT = True
except ImportError:
    REAL_DEVICE_SUPPORT = False

class Physics:
    def __init__(self, hardware_version=3):
        """
        If a real Haply device is found, use it. Otherwise, fallback to sim.
        """
        self.hardware_version = hardware_version
        self.device_present = False

        if REAL_DEVICE_SUPPORT:
            self.port = self.find_device_port()
            if self.port:
                print(f"[PHYSICS] Found device on {self.port}")
                self.device_present = True
            else:
                print("[PHYSICS] No device found; simulating haptics with mouse.")
        else:
            print("[PHYSICS] No serial library installed; simulating haptics with mouse.")
            self.port = None

    def find_device_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "Arduino Zero" in p.description:
                return p.device
        return None

    def is_device_connected(self):
        return self.device_present

    def update_force(self, force_vector):
        """
        If real device is connected, send (fx, fy) -> device motor torques.
        Otherwise, do nothing.
        """
        if self.device_present:
            pass  # handle real device
        else:
            pass

    def close(self):
        print("[PHYSICS] Closed.")


###############################################################################
# Minimal GRAPHICS class (similar to PA1) with two panels
###############################################################################
class Graphics:
    def __init__(self, device_connected, window_size=(1200,600)):
        pygame.init()
        self.window_size = window_size
        self.window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Drone Dynamics - 2 Panels (Mouse/Haptic Control)")

        # Create two surfaces: left for handle, right for drone
        self.surface_left  = pygame.Surface((window_size[0]//2, window_size[1]))
        self.surface_right = pygame.Surface((window_size[0]//2, window_size[1]))

        self.clock = pygame.time.Clock()
        self.FPS = 100
        self.font = pygame.font.Font(None, 24)

        self.device_connected = device_connected

        # Pseudo-haptics parameters if no device
        self.sim_k = 0.4
        self.sim_b = 0.8
        # We won't do large scale factors for "handle side" movement
        self.handle_scale = 1.0

        # For the drone side, we can do standard "pixels per meter"
        self.drone_scale = 300  # px per meter

        # Colors
        self.lightblue = (200,200,255)
        self.white     = (255,255,255)
        self.black     = (0,0,0)
        self.red       = (255,0,0)
        self.green     = (0,255,0)

    def get_events(self):
        """
        Return (keyups, mouse_pos)
        """
        events = pygame.event.get()
        keyups = []
        for e in events:
            if e.type == pygame.QUIT:
                sys.exit(0)
            elif e.type == pygame.KEYUP:
                keyups.append(e.key)
        mouse_pos = pygame.mouse.get_pos()
        return keyups, mouse_pos

    def erase_surfaces(self):
        self.surface_left.fill(self.white)
        self.surface_right.fill(self.lightblue)

    def sim_forces(self, handle_pos, external_force, mouse_pos):
        """
        If no real device, we simulate a handle connected by a spring to the mouse,
        plus a damping effect from the external force.
        """
        diff = (mouse_pos[0] - handle_pos[0], mouse_pos[1] - handle_pos[1])
        # Let's skip a fancy scale for external force for the handle
        scaled_force = (external_force[0]*0.05, external_force[1]*0.05)

        k_term = (self.sim_k * diff[0], self.sim_k * diff[1])
        b_term = (scaled_force[0]/self.sim_b, scaled_force[1]/self.sim_b)
        dp = (k_term[0] - b_term[0], k_term[1] - b_term[1])
        new_handle = (handle_pos[0] + dp[0], handle_pos[1] + dp[1])
        return new_handle

    def render_left_panel(self, handle_pos, external_force):
        """
        Draw the handle in the left surface
        """
        # color depends on force magnitude
        fm = math.hypot(external_force[0], external_force[1])
        col_val = min(255, int(50 + fm*10))
        color = (255, col_val, col_val)

        # handle is a rectangle
        rect = pygame.Rect(0,0,40,40)
        rect.center = (int(handle_pos[0]), int(handle_pos[1]))
        pygame.draw.rect(self.surface_left, color, rect, border_radius=8)

        # Optionally, draw center crosshair
        w,h = self.surface_left.get_size()
        cx, cy = w//2, h//2
        pygame.draw.line(self.surface_left, self.black, (cx-10,cy), (cx+10,cy),2)
        pygame.draw.line(self.surface_left, self.black, (cx,cy-10), (cx,cy+10),2)

    def convert_drone_to_screen(self, x_m, y_m):
        """
        Convert drone world coords (meters) to right-surface pixel coords
        """
        w2,h2 = self.surface_right.get_size()
        cx, cy = w2//2, h2//2
        sx = cx + x_m*self.drone_scale
        sy = cy - y_m*self.drone_scale
        return (int(sx), int(sy))

    def render_right_panel(self, drone_pos, drone_radius):
        """
        Draw the drone in the right surface
        """
        c = self.convert_drone_to_screen(drone_pos[0], drone_pos[1])
        rad_px = int(drone_radius*self.drone_scale)
        pygame.draw.circle(self.surface_right, self.red, c, rad_px)

        # Optionally, draw crosshair in the center as well
        w2,h2 = self.surface_right.get_size()
        cx, cy = w2//2, h2//2
        pygame.draw.line(self.surface_right, self.black, (cx-10,cy), (cx+10,cy),2)
        pygame.draw.line(self.surface_right, self.black, (cx,cy-10), (cx,cy+10),2)

    def finalize(self):
        """
        Blit the left and right surfaces onto the main window
        """
        self.window.blit(self.surface_left, (0,0))
        self.window.blit(self.surface_right, (self.window_size[0]//2,0))
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()


###############################################################################
# The main DRONE class to tie it all together
###############################################################################
class DroneDynamicsTwoPanels:
    def __init__(self):
        self.physics = Physics(hardware_version=3)
        self.graphics = Graphics(self.physics.is_device_connected(), (1200,600))

        # We'll do a simple mass-based drone dynamic
        self.drone_pos = np.array([0.0,0.0], dtype=float)
        self.drone_vel = np.array([0.0,0.0], dtype=float)
        self.drone_radius = 0.03  # in meters, can be changed

        self.mass = 1.0
        self.damping = 0.95

        # The handle/haptic side
        # We'll store a position for the handle in left panel
        wleft,hleft = self.graphics.surface_left.get_size()
        self.handle_pos = np.array([wleft//2, hleft//2], dtype=float)  # start in center
        self.mouse_pos  = np.array([wleft//2, hleft//2], dtype=float)

        # time
        self.last_time = time.time()

    def run(self):
        try:
            while True:
                self.loop_once()
        finally:
            self.close()

    def close(self):
        self.graphics.close()
        self.physics.close()

    def loop_once(self):
        now = time.time()
        dt = now - self.last_time
        if dt>0.05:
            dt = 0.05
        self.last_time = now

        keyups, mp = self.graphics.get_events()
        for k in keyups:
            if k == ord('q'):
                sys.exit(0)
            elif k == ord('r'):
                # reset
                self.drone_pos[:] = 0.0
                self.drone_vel[:] = 0.0
                wleft,hleft = self.graphics.surface_left.get_size()
                self.handle_pos[:] = [wleft//2, hleft//2]

        self.mouse_pos = np.array(mp)

        # 1) We want to compute a force that we apply to the drone
        #    based on the handle position relative to the center
        wleft,hleft = self.graphics.surface_left.get_size()
        cx, cy = wleft//2, hleft//2
        dx = (self.handle_pos[0] - cx)
        dy = (self.handle_pos[1] - cy)
        # scale
        scale_factor = 0.01  # This sets how strong the user control is
        user_fx = dx*scale_factor
        user_fy = -(dy*scale_factor)  # invert y for typical coordinate sense
        user_force = np.array([user_fx,user_fy])

        # 2) integrate the drone (mass-damper)
        ax = user_force[0]/self.mass
        ay = user_force[1]/self.mass
        self.drone_vel[0]+= ax*dt
        self.drone_vel[1]+= ay*dt
        self.drone_vel*= self.damping
        self.drone_pos+= self.drone_vel*dt

        # 3) maybe we want to produce some haptic feedback = e.g. we can set none or
        #    just echo the user force
        haptic_force = -user_force  # negative feedback
        if self.physics.is_device_connected():
            self.physics.update_force(haptic_force)
        else:
            new_handle = self.graphics.sim_forces(self.handle_pos, haptic_force, self.mouse_pos)
            self.handle_pos[:] = new_handle

        # 4) draw
        self.graphics.erase_surfaces()
        # left panel
        self.graphics.render_left_panel(self.handle_pos, haptic_force)
        # right panel
        self.graphics.render_right_panel(self.drone_pos, self.drone_radius)
        # finalize
        self.graphics.finalize()

if __name__=="__main__":
    app = DroneDynamicsTwoPanels()
    app.run()
