# drone_env_gaussian_wind.py
# -------------------------------------------------------------------
# A 2D drone environment with:
#  - Two-panel display (left= handle, right= larger environment)
#  - Constrained handle
#  - Many trees, collisions with threshold for both trees and walls
#  - Gaussian-based wind with smaller variance, limited direction range
#  - The handle block color is a gradient from light red to deep red
#    depending on net force magnitude.

import sys
import math
import time
import numpy as np
import pygame
import random

###############################################################################
# PHYSICS
###############################################################################
try:
    import serial.tools.list_ports
    REAL_DEVICE_SUPPORT = True
except ImportError:
    REAL_DEVICE_SUPPORT = False

class Physics:
    def __init__(self, hardware_version=3):
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
            print("[PHYSICS] No serial library; simulating.")
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
        if self.device_present:
            # convert (fx,fy)-> device
            pass
        else:
            pass

    def close(self):
        print("[PHYSICS] Closed.")


###############################################################################
# GRAPHICS: two surfaces
###############################################################################
class Graphics:
    def __init__(self, device_connected, window_size=(1200,600)):
        pygame.init()
        self.window_size = window_size
        self.window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Drone Env with Gaussian Wind & Collision Threshold")

        self.surface_left = pygame.Surface((window_size[0]//2, window_size[1]))
        self.surface_right= pygame.Surface((window_size[0]//2, window_size[1]))

        self.clock = pygame.time.Clock()
        self.FPS = 100
        self.font = pygame.font.Font(None, 24)

        self.device_connected = device_connected
        # Pseudo-haptics
        self.sim_k = 0.4
        self.sim_b = 0.8

        # handle constraints
        wL,hL= self.surface_left.get_size()
        self.handle_min_x= 50
        self.handle_max_x= wL-50
        self.handle_min_y= 50
        self.handle_max_y= hL-50

        # drone scale
        self.drone_scale= 300

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
        # pseudo-haptics with clamp
        diff = (mouse_pos[0]-handle_pos[0], mouse_pos[1]-handle_pos[1])
        scale_f = (external_force[0]*0.05, external_force[1]*0.05)
        k_term = (self.sim_k*diff[0], self.sim_k*diff[1])
        b_term = (scale_f[0]/self.sim_b, scale_f[1]/self.sim_b)
        dp = (k_term[0]-b_term[0], k_term[1]-b_term[1])

        new_x= handle_pos[0]+ dp[0]
        new_y= handle_pos[1]+ dp[1]
        # clamp
        if new_x< self.handle_min_x: new_x= self.handle_min_x
        if new_x> self.handle_max_x: new_x= self.handle_max_x
        if new_y< self.handle_min_y: new_y= self.handle_min_y
        if new_y> self.handle_max_y: new_y= self.handle_max_y

        return (new_x, new_y)

    def convert_drone_to_screen(self, x_m, y_m):
        w2,h2 = self.surface_right.get_size()
        cx,cy = w2//2, h2//2
        sx= cx + x_m*self.drone_scale
        sy= cy - y_m*self.drone_scale
        return (int(sx), int(sy))

    def render_left(self, handle_pos, total_force):
        """
        Color: from light red to deep red depending on force magnitude
        We'll define a max range for the color, e.g. up to 2.0
        """
        fm= math.hypot(total_force[0], total_force[1])
        # ratio in [0,1]
        ratio= min(1.0, fm/2.0)
        # We start from (255, 200, 200) for ratio=0 to (255, 0, 0) for ratio=1
        # So the green/blue channels go from 200 down to 0
        gb= int(200*(1-ratio))
        color= (255, gb, gb)

        rect= pygame.Rect(0,0,40,40)
        rect.center= (int(handle_pos[0]), int(handle_pos[1]))
        pygame.draw.rect(self.surface_left, color, rect, border_radius=6)

        # crosshair + instructions
        wL,hL= self.surface_left.get_size()
        cx,cy= wL//2, hL//2
        pygame.draw.line(self.surface_left, self.black, (cx-10,cy), (cx+10,cy),2)
        pygame.draw.line(self.surface_left, self.black, (cx,cy-10), (cx,cy+10),2)

        lines= [
            "Keys:",
            " Q=quit, R=reset, W=toggle wind",
            "Left: handle (mouse/haptic).",
            "Right: drone in bigger env, walls=box, trees=green.",
            "Collision with threshold. Gaussian wind. Force color => handle load."
        ]
        yoff=10
        for ln in lines:
            surf= self.font.render(ln, True, (0,0,0))
            self.surface_left.blit(surf, (10,yoff))
            yoff+=20

    def render_right(self, drone_pos, drone_radius, walls, trees, collision_count,
                     wind_vec, wind_on):
        (xmin,xmax,ymin,ymax)= walls
        # draw walls
        c_tl= self.convert_drone_to_screen(xmin, ymax)
        c_br= self.convert_drone_to_screen(xmax, ymin)
        rect= pygame.Rect(c_tl[0], c_tl[1], c_br[0]-c_tl[0], c_br[1]-c_tl[1])
        pygame.draw.rect(self.surface_right, self.black, rect,2)

        # draw trees
        for (tx,ty,tr) in trees:
            cc= self.convert_drone_to_screen(tx,ty)
            rp= int(tr*self.drone_scale)
            pygame.draw.circle(self.surface_right, self.green, cc, rp)

        # drone
        cdrone= self.convert_drone_to_screen(drone_pos[0], drone_pos[1])
        rpix= int(drone_radius*self.drone_scale)
        pygame.draw.circle(self.surface_right, self.red, cdrone, rpix)

        # collisions
        t_surf= self.font.render(f"Collisions={collision_count}", True, (0,0,0))
        self.surface_right.blit(t_surf, (10,10))

        # wind arrow if on
        if wind_on:
            wmag= np.linalg.norm(wind_vec)
            if wmag>1e-3:
                angle= math.atan2(wind_vec[1], wind_vec[0])
                arrow_len= 0.2*wmag
                start= cdrone
                end= (cdrone[0]+ arrow_len*self.drone_scale*math.cos(angle),
                      cdrone[1]- arrow_len*self.drone_scale*math.sin(angle))
                pygame.draw.line(self.surface_right, (0,0,255), start, end,3)

    def finalize(self):
        self.window.blit(self.surface_left,(0,0))
        self.window.blit(self.surface_right,(self.window_size[0]//2,0))
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()


###############################################################################
# Main Drone environment
###############################################################################
class DroneEnvGaussWind:
    def __init__(self):
        self.physics= Physics(hardware_version=3)
        self.graphics= Graphics(self.physics.is_device_connected(), (1200,600))

        # handle
        wL,hL= self.graphics.surface_left.get_size()
        self.handle_pos= np.array([wL//2, hL//2],dtype=float)
        self.mouse_pos= self.handle_pos.copy()

        # environment
        self.xmin,self.xmax= -1.0, 1.0
        self.ymin,self.ymax= -0.8, 0.8

        # drone
        self.drone_pos= np.array([0.0,0.0],dtype=float)
        self.drone_vel= np.array([0.0,0.0],dtype=float)
        self.drone_radius= 0.03
        self.mass=1.0
        self.damping=0.96

        # collisions
        self.collision_count= 0
        self.colliding_set_walls= False  # track if currently colliding with walls
        self.colliding_trees= set()
        self.leave_threshold= 0.01  # must move away by 1 cm

        # trees
        self.num_trees= 12
        self.tree_min_size=0.02
        self.tree_max_size=0.07
        self.trees=[]
        self.init_trees()

        # wind
        self.wind_on= False
        # We'll define a seeded random
        self.rng= random.Random(9999)
        # define initial heading + 45 deg range
        self.init_dir_deg= 30.0
        self.dir_range_deg= 90.0  # total range => ±45 deg
        self.wind_mag_min= 0.0
        self.wind_mag_max= 0.5  # lower max for smaller variance
        self.gauss_sigma=0.05   # small standard dev for each step
        # we'll store a list of frames again
        self.num_wind_steps= 3000
        self.wind_data=[]
        self.wind_idx=0
        # generate
        self.generate_wind_list()

        # collision bump
        self.bump_force= np.array([0.0,0.0],dtype=float)
        self.bump_ttl=0.0
        self.bump_decay=1.5

        self.last_time= time.time()

    def init_trees(self):
        random.seed(1234)
        for i in range(self.num_trees):
            x= random.uniform(self.xmin+0.1, self.xmax-0.1)
            y= random.uniform(self.ymin+0.1, self.ymax-0.1)
            r= random.uniform(self.tree_min_size, self.tree_max_size)
            self.trees.append( (x,y,r) )

    def generate_wind_list(self):
        """
        We'll maintain a 'current' angle & magnitude that evolves with small gaussian steps,
        but clamp angle in [init_dir ± 45 deg], and magnitude in [wind_mag_min, wind_mag_max].
        """
        dir_rad= math.radians(self.init_dir_deg)
        half_range= math.radians(self.dir_range_deg/2.0)
        mag= (self.wind_mag_min + self.wind_mag_max)*0.5  # mid
        angle= dir_rad

        for i in range(self.num_wind_steps):
            # random small step in magnitude
            dm= self.rng.gauss(0.0, self.gauss_sigma)
            mag+= dm
            mag= max(self.wind_mag_min, min(self.wind_mag_max, mag))

            # random small step in angle
            dtheta= self.rng.gauss(0.0, self.gauss_sigma)
            angle+= dtheta
            # clamp angle to [dir_rad-half_range, dir_rad+half_range]
            lo= dir_rad- half_range
            hi= dir_rad+ half_range
            if angle< lo: angle= lo
            if angle> hi: angle= hi

            self.wind_data.append((mag, angle))
        print("[INFO] wind_data created with length", len(self.wind_data))

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
        now= time.time()
        dt= now- self.last_time
        if dt>0.05:
            dt=0.05
        self.last_time= now

        keyups, mp= self.graphics.get_events()
        for k in keyups:
            if k== ord('q'):
                sys.exit(0)
            elif k== ord('r'):
                self.reset_env()
            elif k== ord('w'):
                self.wind_on= not self.wind_on
                if not self.wind_on:
                    self.wind_idx=0
                print("[INFO] Toggled wind =>", self.wind_on)

        self.mouse_pos= np.array(mp)

        # user force
        f_user= self.compute_user_force()
        # wind
        f_wind= np.array([0.0,0.0])
        if self.wind_on:
            mag,angle= self.wind_data[self.wind_idx]
            self.wind_idx= (self.wind_idx+1)% len(self.wind_data)
            fx= mag*math.cos(angle)
            fy= mag*math.sin(angle)
            f_wind= np.array([fx,fy],dtype=float)

        self.update_bump(dt)
        fx= f_user[0]+ f_wind[0]+ self.bump_force[0]
        fy= f_user[1]+ f_wind[1]+ self.bump_force[1]

        # integrate drone
        ax= fx/self.mass
        ay= fy/self.mass
        self.drone_vel[0]+= ax*dt
        self.drone_vel[1]+= ay*dt
        self.drone_vel*= self.damping
        self.drone_pos+= self.drone_vel*dt

        # check wall collisions with threshold
        self.check_wall_collision_threshold()

        # check tree collisions with threshold
        self.check_tree_collision_threshold()

        # build haptic force => negative wind + negative collision bump
        hf_x= -f_wind[0] - self.bump_force[0]
        hf_y= -f_wind[1] - self.bump_force[1]
        haptic_force= np.array([hf_x, hf_y], dtype=float)

        # real or pseudo
        if self.physics.is_device_connected():
            self.physics.update_force(haptic_force)
        else:
            newpos= self.graphics.sim_forces(self.handle_pos, haptic_force, self.mouse_pos)
            self.handle_pos[:]= newpos

        # draw
        self.graphics.erase_surfaces()
        net_fx= fx
        net_fy= fy
        self.graphics.render_left(self.handle_pos, (net_fx, net_fy))
        walls= (self.xmin, self.xmax, self.ymin, self.ymax)
        self.graphics.render_right(self.drone_pos, self.drone_radius, walls,
                                   self.trees, self.collision_count,
                                   (f_wind[0], f_wind[1]), self.wind_on)
        self.graphics.finalize()

    def compute_user_force(self):
        """ scale_factor=0.01 from handle offset to user force """
        wL,hL= self.graphics.surface_left.get_size()
        cx,cy= wL//2, hL//2
        dx= self.handle_pos[0]- cx
        dy= self.handle_pos[1]- cy
        scale_factor=0.01
        fx= dx*scale_factor
        fy= -dy*scale_factor
        return np.array([fx,fy],dtype=float)

    def check_wall_collision_threshold(self):
        """
        We track if currently colliding with the wall or not (colliding_set_walls).
        We only count a new collision if we were not previously in collision.
        Once in collision, we remain so until we are > leave_threshold from the boundary.
        """
        in_collision= self.is_in_wall_collision()
        if in_collision:
            if not self.colliding_set_walls:
                # new collision
                self.collision_count+=1
                self.trigger_bump()
                self.colliding_set_walls= True
        else:
            if self.colliding_set_walls:
                # see if we've left by threshold
                # we define "left by threshold" as being more than threshold away from any wall
                distw= self.dist_to_wall()
                if distw> self.leave_threshold:
                    self.colliding_set_walls= False

    def is_in_wall_collision(self):
        # if the drone is out of [xmin+radius, xmax-radius], etc. then it's "colliding"
        # or very slightly within the boundary
        if self.drone_pos[0]< self.xmin+self.drone_radius:
            # push out
            self.drone_pos[0]= self.xmin+self.drone_radius
            self.drone_vel[0]*= -0.5
            return True
        if self.drone_pos[0]> self.xmax-self.drone_radius:
            self.drone_pos[0]= self.xmax-self.drone_radius
            self.drone_vel[0]*= -0.5
            return True
        if self.drone_pos[1]< self.ymin+self.drone_radius:
            self.drone_pos[1]= self.ymin+self.drone_radius
            self.drone_vel[1]*= -0.5
            return True
        if self.drone_pos[1]> self.ymax-self.drone_radius:
            self.drone_pos[1]= self.ymax-self.drone_radius
            self.drone_vel[1]*= -0.5
            return True
        return False

    def dist_to_wall(self):
        """
        Return how far the drone is from the nearest boundary.
        If > threshold => not colliding
        """
        dx_left= self.drone_pos[0] - (self.xmin+self.drone_radius)
        dx_right= (self.xmax-self.drone_radius) - self.drone_pos[0]
        dy_bottom= self.drone_pos[1] - (self.ymin+self.drone_radius)
        dy_top= (self.ymax-self.drone_radius) - self.drone_pos[1]
        # The min of these is how close we are to a boundary
        return min(dx_left, dx_right, dy_bottom, dy_top)

    def check_tree_collision_threshold(self):
        """
        If dist< sumr => collision if not in colliding_trees
        Must leave sumr+ threshold to remove from colliding_trees.
        """
        for i,(tx,ty,tr) in enumerate(self.trees):
            dx= self.drone_pos[0]- tx
            dy= self.drone_pos[1]- ty
            dist= math.hypot(dx,dy)
            sumr= self.drone_radius+ tr
            if dist< sumr:
                if i not in self.colliding_trees:
                    # new collision
                    # push out
                    overlap= sumr- dist
                    if dist>1e-6:
                        nx= dx/dist
                        ny= dy/dist
                        self.drone_pos[0]+= nx*overlap
                        self.drone_pos[1]+= ny*overlap
                    else:
                        self.drone_pos[0]+=0.001
                    self.drone_vel*= -0.5
                    self.collision_count+=1
                    self.trigger_bump()
                    self.colliding_trees.add(i)
            else:
                if i in self.colliding_trees:
                    # we must see if dist> sumr+ threshold
                    if dist> sumr+ self.leave_threshold:
                        self.colliding_trees.remove(i)

    def trigger_bump(self):
        """
        short haptic bump
        """
        mag=0.5
        ang= random.random()*2*math.pi
        self.bump_force= np.array([mag*math.cos(ang), mag*math.sin(ang)])
        self.bump_ttl=0.5

    def update_bump(self, dt):
        if self.bump_ttl>0:
            self.bump_ttl-= dt
            if self.bump_ttl<=0:
                self.bump_ttl=0
                self.bump_force[:]=0
            else:
                fade= math.exp(-self.bump_decay*dt)
                self.bump_force*= fade

    def reset_env(self):
        wL,hL= self.graphics.surface_left.get_size()
        self.handle_pos[:]= [wL//2,hL//2]
        self.drone_pos[:]=0
        self.drone_vel[:]=0
        self.collision_count=0
        self.colliding_set_walls= False
        self.colliding_trees.clear()
        self.bump_force[:]=0
        self.bump_ttl=0.0
        self.wind_on= False
        self.wind_idx=0

if __name__=="__main__":
    env= DroneEnvGaussWind()
    env.run()
