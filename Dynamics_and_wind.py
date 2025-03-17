# drone_env.py
# ---------------------------------------------------------------
# A 2D drone dynamic model with:
#  - Two-panel display (left= handle, right= drone world)
#  - Full constraints on the handle in left panel
#  - Walled environment in right panel + collision counting
#  - Random trees (green circles) with adjustable size range
#  - Wind that can be toggled via "w", also fed into haptic feedback

import sys
import math
import time
import numpy as np
import pygame
import random

###############################################################################
# Minimal PHYSICS class
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
                print("[PHYSICS] No device found; simulating.")
        else:
            print("[PHYSICS] Serial library not installed; simulating.")
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
        If real device is connected, convert (fx,fy)-> motor torques.
        Otherwise do nothing.
        """
        if self.device_present:
            pass
        else:
            pass

    def close(self):
        print("[PHYSICS] Closed.")


###############################################################################
# Minimal GRAPHICS class with 2 surfaces (left=handle, right=drone world)
###############################################################################
class Graphics:
    def __init__(self, device_connected, window_size=(1200,600)):
        pygame.init()
        self.window_size = window_size
        self.window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Drone Env with Walls, Trees, Wind")

        self.surface_left = pygame.Surface((window_size[0]//2, window_size[1]))
        self.surface_right= pygame.Surface((window_size[0]//2, window_size[1]))

        self.clock = pygame.time.Clock()
        self.FPS = 100

        self.font = pygame.font.Font(None, 24)

        self.device_connected = device_connected

        # For pseudo-haptics if no device
        self.sim_k = 0.4
        self.sim_b = 0.8

        # We'll define handle constraints based on typical haptic device range
        # from older PA1. Let's define a rectangle in the left surface
        wL, hL = self.surface_left.get_size()
        self.handle_min_x = 50
        self.handle_max_x = wL - 50
        self.handle_min_y = 50
        self.handle_max_y = hL - 50

        # Drone scale
        self.drone_scale = 300  # px per meter

        # Colors
        self.white     = (255,255,255)
        self.lightblue = (200,200,255)
        self.black     = (0,0,0)
        self.red       = (255,0,0)
        self.green     = (0,180,0)

    def get_events(self):
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
        Pseudo-haptics: handle connected by a spring to the mouse,
        plus damping from external_force. Then clamp handle pos in all 4 directions.
        """
        diff = (mouse_pos[0] - handle_pos[0], mouse_pos[1] - handle_pos[1])
        # let's scale external force a bit so user sees effect
        scaled_force = (external_force[0]*0.05, external_force[1]*0.05)
        k_term = (self.sim_k * diff[0], self.sim_k * diff[1])
        b_term = (scaled_force[0]/self.sim_b, scaled_force[1]/self.sim_b)
        dp = (k_term[0] - b_term[0], k_term[1] - b_term[1])

        new_x = handle_pos[0] + dp[0]
        new_y = handle_pos[1] + dp[1]

        # clamp
        if new_x< self.handle_min_x: new_x= self.handle_min_x
        if new_x> self.handle_max_x: new_x= self.handle_max_x
        if new_y< self.handle_min_y: new_y= self.handle_min_y
        if new_y> self.handle_max_y: new_y= self.handle_max_y

        return (new_x, new_y)

    def convert_drone_to_screen(self, x_m, y_m):
        w2,h2 = self.surface_right.get_size()
        cx,cy = w2//2, h2//2
        sx = cx + x_m*self.drone_scale
        sy = cy - y_m*self.drone_scale
        return (int(sx), int(sy))

    def render_left(self, handle_pos, total_force):
        """
        Draw the handle rectangle. Color depends on magnitude of total_force.
        """
        mag = math.hypot(total_force[0], total_force[1])
        cval = min(255, int(40 + mag*10))
        color = (255,cval,cval)
        rect = pygame.Rect(0,0,40,40)
        rect.center = (int(handle_pos[0]), int(handle_pos[1]))
        pygame.draw.rect(self.surface_left, color, rect, border_radius=8)

        # draw reference cross in center
        wL,hL = self.surface_left.get_size()
        cx, cy = wL//2, hL//2
        pygame.draw.line(self.surface_left, self.black, (cx-10, cy), (cx+10,cy),2)
        pygame.draw.line(self.surface_left, self.black, (cx,cy-10), (cx,cy+10),2)

    def render_right(self, drone_pos, drone_radius, walls, collisions,
                     trees, wind_on, wind_vec):
        """
        Draw the drone, the boundary walls, and the trees in the right surface.
        - walls is (xmin, xmax, ymin, ymax).
        - collisions is how many collisions have happened (for debug).
        - trees is a list of (x,y, r).
        - wind_on is bool, wind_vec is the current wind vector (for optional arrow).
        """
        # 1) draw walls
        (xmin, xmax, ymin, ymax) = walls
        w2,h2 = self.surface_right.get_size()
        # convert corners
        c_tl = self.convert_drone_to_screen(xmin, ymax)
        c_br = self.convert_drone_to_screen(xmax, ymin)
        rect = pygame.Rect(c_tl[0], c_tl[1], c_br[0]-c_tl[0], c_br[1]-c_tl[1])
        pygame.draw.rect(self.surface_right, self.black, rect, 2)

        # 2) draw trees
        for (tx,ty,tr) in trees:
            c = self.convert_drone_to_screen(tx, ty)
            rpix = int(tr*self.drone_scale)
            pygame.draw.circle(self.surface_right, self.green, c, rpix)

        # 3) drone
        cdrone = self.convert_drone_to_screen(drone_pos[0], drone_pos[1])
        rpix = int(drone_radius*self.drone_scale)
        pygame.draw.circle(self.surface_right, self.red, cdrone, rpix)

        # optional debug text
        text_surf = self.font.render(f"Collisions={collisions}", True, (0,0,0))
        self.surface_right.blit(text_surf, (10,10))

        # if wind_on => draw arrow
        if wind_on and (abs(wind_vec[0])>1e-3 or abs(wind_vec[1])>1e-3):
            # small arrow
            wmag = math.hypot(wind_vec[0], wind_vec[1])
            angle = math.atan2(wind_vec[1], wind_vec[0])
            arrow_len_m = 0.2*wmag  # scale
            start = cdrone
            end = (cdrone[0] + arrow_len_m*self.drone_scale*math.cos(angle),
                   cdrone[1] - arrow_len_m*self.drone_scale*math.sin(angle))
            pygame.draw.line(self.surface_right, (0,0,255), start, end, 3)

    def finalize(self):
        self.window.blit(self.surface_left, (0,0))
        self.window.blit(self.surface_right,(self.window_size[0]//2,0))
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()


###############################################################################
# Main Drone Env class
###############################################################################
class DroneEnv:
    def __init__(self):
        self.physics = Physics(hardware_version=3)
        self.graphics = Graphics(self.physics.is_device_connected(), (1200,600))

        # handle (left side)
        wL,hL = self.graphics.surface_left.get_size()
        self.handle_pos = np.array([wL//2, hL//2], dtype=float)
        self.mouse_pos  = self.handle_pos.copy()

        # environment bounds in the drone world (meters)
        self.xmin, self.xmax = -0.4, 0.4
        self.ymin, self.ymax = -0.3, 0.3

        # drone
        self.drone_pos = np.array([0.0, 0.0], dtype=float)
        self.drone_vel = np.array([0.0, 0.0], dtype=float)
        self.drone_radius = 0.03
        self.mass = 1.0
        self.damping = 0.96
        self.collision_count = 0

        # random trees
        random.seed(1234)  # consistent each run
        self.num_trees = 5
        self.tree_min_size = 0.02
        self.tree_max_size = 0.05
        self.trees = []
        self.init_trees()

        # wind
        self.wind_on = False
        self.wind_vec = np.array([0.5, 0.2])  # example constant wind, tune as desired

        self.last_time = time.time()

    def init_trees(self):
        """Create random tree positions/sizes inside [xmin,xmax]x[ymin,ymax]."""
        for i in range(self.num_trees):
            x = random.uniform(self.xmin+0.05, self.xmax-0.05)
            y = random.uniform(self.ymin+0.05, self.ymax-0.05)
            r = random.uniform(self.tree_min_size, self.tree_max_size)
            self.trees.append( (x,y,r) )

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
                self.reset_drone()
            elif k == ord('w'):
                self.wind_on = not self.wind_on
                print(f"[INFO] Wind toggled: {self.wind_on}")

        self.mouse_pos = np.array(mp)

        # Step 1: Drone dynamics
        # user force from handle offset
        f_user = self.compute_user_force()
        # wind
        f_wind = np.array([0.0,0.0])
        if self.wind_on:
            f_wind = self.wind_vec.copy()

        total_fx = f_user[0] + f_wind[0]
        total_fy = f_user[1] + f_wind[1]
        # integrate
        ax = total_fx/self.mass
        ay = total_fy/self.mass
        self.drone_vel[0]+= ax*dt
        self.drone_vel[1]+= ay*dt
        self.drone_vel*= self.damping
        self.drone_pos+= self.drone_vel*dt

        # Step 2: check collisions with walls
        collided = self.check_wall_collision()
        if collided:
            self.collision_count+=1

        # Step 3: check collisions with trees
        self.check_trees_collision()

        # Step 4: haptic feedback
        # We'll pass the wind as a negative force to handle, so user "feels" it
        # plus we might add something for collisions or so, but let's keep it simple
        fe = -f_wind

        # For debugging, let's color the handle according to total magnitude
        # i.e. user force + wind => net magnitude
        net_fx = total_fx
        net_fy = total_fy
        net_mag = math.hypot(net_fx, net_fy)

        if self.physics.is_device_connected():
            self.physics.update_force(fe)
        else:
            newpos = self.graphics.sim_forces(self.handle_pos, fe, self.mouse_pos)
            self.handle_pos[:] = newpos

        # Step 5: draw
        self.graphics.erase_surfaces()
        # left
        self.graphics.render_left(self.handle_pos, (net_fx, net_fy))
        # right
        walls = (self.xmin, self.xmax, self.ymin, self.ymax)
        self.graphics.render_right(self.drone_pos, self.drone_radius,
                                   walls, self.collision_count,
                                   self.trees, self.wind_on, self.wind_vec)
        self.graphics.finalize()

    def reset_drone(self):
        """Reset drone + collisions + handle pos."""
        self.drone_pos[:] = 0.0
        self.drone_vel[:] = 0.0
        self.collision_count= 0
        wL,hL = self.graphics.surface_left.get_size()
        self.handle_pos[:] = [wL//2, hL//2]

    def compute_user_force(self):
        """
        The handle offset from center in left panel => user force on drone.
        """
        wL,hL = self.graphics.surface_left.get_size()
        cx, cy = wL//2, hL//2
        dx = self.handle_pos[0] - cx
        dy = self.handle_pos[1] - cy
        scale_factor = 0.01
        fx = dx*scale_factor
        fy = -dy*scale_factor
        return np.array([fx, fy], dtype=float)

    def check_wall_collision(self):
        collided = False
        # clamp
        if self.drone_pos[0] < self.xmin+self.drone_radius:
            self.drone_pos[0] = self.xmin+self.drone_radius
            self.drone_vel[0]*= -0.5  # bounce
            collided= True
        if self.drone_pos[0] > self.xmax-self.drone_radius:
            self.drone_pos[0] = self.xmax-self.drone_radius
            self.drone_vel[0]*= -0.5
            collided= True
        if self.drone_pos[1] < self.ymin+self.drone_radius:
            self.drone_pos[1] = self.ymin+self.drone_radius
            self.drone_vel[1]*= -0.5
            collided= True
        if self.drone_pos[1] > self.ymax-self.drone_radius:
            self.drone_pos[1] = self.ymax-self.drone_radius
            self.drone_vel[1]*= -0.5
            collided= True
        return collided

    def check_trees_collision(self):
        for i,(tx,ty,tr) in enumerate(self.trees):
            dx = self.drone_pos[0] - tx
            dy = self.drone_pos[1] - ty
            dist = math.hypot(dx,dy)
            sumr = self.drone_radius+ tr
            if dist< sumr:
                # collision
                overlap = sumr- dist
                # push out
                if dist>1e-6:
                    nx= dx/dist
                    ny= dy/dist
                    self.drone_pos[0]+= nx* overlap
                    self.drone_pos[1]+= ny* overlap
                else:
                    # degenerate
                    self.drone_pos[0]+= 0.001
                self.drone_vel*= -0.5
                self.collision_count+=1


if __name__=="__main__":
    app = DroneEnv()
    app.run()
