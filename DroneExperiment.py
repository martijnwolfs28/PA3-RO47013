import sys
import math
import time
import numpy as np
import pygame
import random
import pandas as pd
import os

###############################################################################
# Minimal PHYSICS
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
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "Arduino Zero" in p.description:
                return p.device
        return None

    def is_device_connected(self):
        return self.device_present

    def update_force(self, force_vector):
        pass

    def close(self):
        print("[PHYSICS] Closed.")


###############################################################################
# GRAPHICS
###############################################################################
def draw_arrow(surface, start, angle, arrow_len=60, color=(0,0,255), width=3):
    """
    Draw a fixed-length arrow of length `arrow_len` starting at `start`,
    pointing in direction `angle` (radians).
    Includes a small arrowhead.
    """
    end_x = start[0] + arrow_len * math.cos(angle)
    end_y = start[1] - arrow_len * math.sin(angle)

    # Main line
    pygame.draw.line(surface, color, start, (end_x, end_y), width)

    # Arrowhead
    head_len = 10
    head_angle = math.radians(30)  # half-angle of the arrowhead
    left_dir = angle + head_angle
    right_dir = angle - head_angle

    left_tip_x = end_x - head_len * math.cos(left_dir)
    left_tip_y = end_y + head_len * math.sin(left_dir)
    right_tip_x = end_x - head_len * math.cos(right_dir)
    right_tip_y = end_y + head_len * math.sin(right_dir)

    pygame.draw.line(surface, color, (end_x, end_y), (left_tip_x, left_tip_y), width)
    pygame.draw.line(surface, color, (end_x, end_y), (right_tip_x, right_tip_y), width)


class Graphics:
    def __init__(self, device_connected, window_size=(1200,600)):
        pygame.init()
        self.window_size = window_size
        self.window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Drone with Weaker Gradient, Timed Trials, 2.5D Potential")

        self.surface_left = pygame.Surface((window_size[0]//2, window_size[1]))
        self.surface_right= pygame.Surface((window_size[0]//2, window_size[1]))

        self.clock = pygame.time.Clock()
        self.FPS = 100
        self.font = pygame.font.Font(None,24)
        self.device_connected = device_connected

        # Pseudo-haptics
        self.sim_k = 0.4
        self.sim_b = 0.8

        # Handle constraints
        wL,hL = self.surface_left.get_size()
        self.handle_min_x=50
        self.handle_max_x=wL-50
        self.handle_min_y=50
        self.handle_max_y=hL-50

        # Drone scale
        self.drone_scale=300

        # Colors
        self.white=(255,255,255)
        self.lightblue=(200,200,255)
        self.black=(0,0,0)
        self.red=(255,0,0)
        self.green=(0,180,0)

        # Potential surface
        w2,h2= self.surface_right.get_size()
        self.pot_surf=pygame.Surface((w2,h2))

        # We'll let the main class handle environment data

    def get_events(self):
        events=pygame.event.get()
        keyups=[]
        for e in events:
            if e.type==pygame.QUIT:
                sys.exit(0)
            elif e.type==pygame.KEYUP:
                keyups.append(e.key)
        mouse_pos=pygame.mouse.get_pos()
        return keyups, mouse_pos

    def erase_surfaces(self):
        self.surface_left.fill(self.white)

    def sim_forces(self, handle_pos, external_force, mouse_pos):
        """
        Pseudo-haptics: tries to keep handle near the mouse but offset by external_force.
        """
        diff = (mouse_pos[0]-handle_pos[0], mouse_pos[1]-handle_pos[1])
        scaled_f = (external_force[0]*0.05, external_force[1]*0.05)
        k_term = (self.sim_k*diff[0], self.sim_k*diff[1])
        b_term = (scaled_f[0]/self.sim_b, scaled_f[1]/self.sim_b)
        dp = (k_term[0] - b_term[0], k_term[1] - b_term[1])

        new_x= handle_pos[0]+ dp[0]
        new_y= handle_pos[1]+ dp[1]
        if new_x< self.handle_min_x: new_x=self.handle_min_x
        if new_x> self.handle_max_x: new_x=self.handle_max_x
        if new_y< self.handle_min_y: new_y=self.handle_min_y
        if new_y> self.handle_max_y: new_y=self.handle_max_y
        return (new_x,new_y)

    def convert_drone_to_screen(self, x_m, y_m):
        w2,h2= self.surface_right.get_size()
        cx,cy=w2//2,h2//2
        sx= cx + x_m*self.drone_scale
        sy= cy - y_m*self.drone_scale
        return (int(sx),int(sy))

    def render_left(self, handle_pos, total_force, state, wind_on, grad_on,
                    elapsed_time, haptic_wind_mode):
        """
        Left plane: show handle rectangle & text info
        """
        fm=math.hypot(total_force[0], total_force[1])
        ratio= min(1.0, fm/2.0)
        gb=int(200*(1.0-ratio))
        color=(255,gb,gb)

        rect=pygame.Rect(0,0,40,40)
        rect.center=(int(handle_pos[0]), int(handle_pos[1]))
        pygame.draw.rect(self.surface_left, color, rect, border_radius=6)

        wL,hL=self.surface_left.get_size()
        cx,cy=wL//2,hL//2
        pygame.draw.line(self.surface_left,self.black,(cx-10,cy),(cx+10,cy),2)
        pygame.draw.line(self.surface_left,self.black,(cx,cy-10),(cx,cy+10),2)

        lines=[
            f"STATE: {state}",
            f"Time: {elapsed_time:.2f}s",
            f"Wind: {'ON' if wind_on else 'OFF'}",
            f"Grad: {'ON' if grad_on else 'OFF'}",
            f"HapticWindMode: {haptic_wind_mode}",
            "",
            "Keys: Q=quit(no save), R=reset(no save)",
            " W=wind, G=gradient, Z=toggle wind mode",
            " S=start, E=end+save, V=pot map",
            " D=toggle wind arrow display"
        ]
        yoff=10
        for ln in lines:
            srf=self.font.render(ln, True,(0,0,0))
            self.surface_left.blit(srf,(10,yoff))
            yoff+=20

    def render_right(self, run_number, total_runs,
                     drone_pos, drone_radius, walls, trees, collision_count,
                     wind_vec, wind_on, gradient_on,
                     start_area, finish_area,
                     pot_map_on, pot_surf,
                     wind_in_corner=False,
                     show_start_message=False,
                     show_end_message=False,
                     show_done_message=False):
        """
        Right plane: draws environment walls, trees, drone, wind arrow, etc.
        Also draws messages for run number, start/end prompts, etc.
        """
        # pot map or normal fill
        if pot_map_on:
            self.surface_right.blit(pot_surf,(0,0))
        else:
            self.surface_right.fill(self.lightblue)

        (xmin,xmax,ymin,ymax)= walls
        c_tl=self.convert_drone_to_screen(xmin,ymax)
        c_br=self.convert_drone_to_screen(xmax,ymin)
        rect=pygame.Rect(c_tl[0], c_tl[1], c_br[0]-c_tl[0], c_br[1]-c_tl[1])
        pygame.draw.rect(self.surface_right,(0,0,0), rect,2)

        # start area
        s_tl=self.convert_drone_to_screen(start_area[0], start_area[3])
        s_br=self.convert_drone_to_screen(start_area[2], start_area[1])
        start_rect= pygame.Rect(s_tl[0], s_tl[1], s_br[0]-s_tl[0], s_br[1]-s_tl[1])
        pygame.draw.rect(self.surface_right,(0,150,0),start_rect,2)

        # finish area
        f_tl=self.convert_drone_to_screen(finish_area[0], finish_area[3])
        f_br=self.convert_drone_to_screen(finish_area[2], finish_area[1])
        finish_rect= pygame.Rect(f_tl[0], f_tl[1], f_br[0]-f_tl[0], f_br[1]-f_tl[1])
        pygame.draw.rect(self.surface_right,(150,0,150), finish_rect,2)

        # trees
        for (tx,ty,tr) in trees:
            cc= self.convert_drone_to_screen(tx,ty)
            rp=int(tr*300)
            pygame.draw.circle(self.surface_right,(0,180,0), cc, rp)

        # drone
        cdrone= self.convert_drone_to_screen(drone_pos[0],drone_pos[1])
        rpix=int(drone_radius*300)
        pygame.draw.circle(self.surface_right,(255,0,0), cdrone, rpix)

        # collisions
        t_surf=self.font.render(f"Collisions={collision_count}",True,(0,0,0))
        self.surface_right.blit(t_surf,(10,10))

        #######################################################################
        # NEW: Show run number in top center
        #######################################################################
        w2,h2= self.surface_right.get_size()
        run_text = f"Run {run_number} / {total_runs}"
        run_surf = self.font.render(run_text, True, (0,0,0))
        run_rect = run_surf.get_rect(midtop=(w2//2, 10))
        self.surface_right.blit(run_surf, run_rect)
        #######################################################################

        # wind arrow
        if wind_on:
            wmag = math.hypot(wind_vec[0], wind_vec[1])
            if wmag > 1e-3:
                angle = math.atan2(wind_vec[1], wind_vec[0])
                if not wind_in_corner:
                    arrow_len = 0.2 * wmag
                    end = (cdrone[0] + arrow_len*300*math.cos(angle),
                           cdrone[1] - arrow_len*300*math.sin(angle))
                    pygame.draw.line(self.surface_right, (0,0,255), cdrone, end, 3)
                else:
                    corner_pos = (50, 50)
                    draw_arrow(self.surface_right, corner_pos, angle,
                               arrow_len=60, color=(0,0,255), width=3)
                    mag_str = f"{wmag:.1f}"
                    txt = self.font.render(f"Wind: {mag_str}", True, (0,0,0))
                    self.surface_right.blit(txt, (corner_pos[0], corner_pos[1]+20))

        #######################################################################
        # NEW: show start/end/done messages in center of right plane
        #######################################################################
        if show_start_message:
            msg = "Press 'S' to start"
            s_surf = self.font.render(msg, True, (0,0,0))
            s_rect = s_surf.get_rect(center=(w2//2, h2//2))
            self.surface_right.blit(s_surf, s_rect)

        if show_end_message:
            msg_end = "Endpoint reached, press 'E' to simulate next environment"
            e_surf = self.font.render(msg_end, True, (0,0,0))
            e_rect = e_surf.get_rect(center=(w2//2, h2//2))
            self.surface_right.blit(e_surf, e_rect)

        if show_done_message:
            msg_done = "Experiment done, press 'Q' to quit"
            d_surf = self.font.render(msg_done, True, (0,0,0))
            d_rect = d_surf.get_rect(center=(w2//2, h2//2 + 40))
            self.surface_right.blit(d_surf, d_rect)
        #######################################################################

    def finalize(self):
        self.window.blit(self.surface_left,(0,0))
        self.window.blit(self.surface_right,(self.window_size[0]//2,0))
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()


###############################################################################
# MAIN DRONE CLASS
###############################################################################
class DroneWithBorderGradient:
    def __init__(self):
        self.physics=Physics(hardware_version=3)
        self.graphics=Graphics(self.physics.is_device_connected(),(1200,600))

        # Constants
        self.TOTAL_RUNS = 10  # total environments/runs

        # handle
        wL,hL=self.graphics.surface_left.get_size()
        self.handle_pos=np.array([wL//2,hL//2],dtype=float)
        self.mouse_pos=self.handle_pos.copy()

        # environment bounding box
        self.xmin,self.xmax= -1.0,1.0
        self.ymin,self.ymax= -0.8,0.8

        # start / finish areas (these will get changed each run depending on odd/even)
        self.start_area = [0,0,0,0]
        self.finish_area= [0,0,0,0]

        # drone
        
        self.drone_vel=np.array([0.0,0.0],dtype=float)
        self.drone_radius=0.03
        self.mass=1.0
        self.damping=0.96

        # collisions
        self.collision_count=0
        self.wall_collision=False
        self.tree_collision_set=set()
        self.leave_threshold=0.01

        # tree parameters
        self.num_trees=15
        self.tree_min_size=0.02
        self.tree_max_size=0.07
        self.trees=[]  # current environment's trees

        # wind
        self.wind_on=False
        self.rng=random.Random(3333)
        self.num_wind_steps=2000
        self.wind_data=[]
        self.wind_idx=0
        self.init_dir_deg=30.0
        self.dir_range_deg=90.0
        self.wind_mag_min=0.0
        self.wind_mag_max=1.0
        self.gauss_sigma=0.05
        self.generate_wind_list()

        # collision bump
        self.bump_force=np.array([0.0,0.0],dtype=float)
        self.bump_ttl=0.0
        self.bump_decay=1.5

        # gradient
        self.gradient_on=False
        self.repulse_const=0.005

        # states
        self.state="IDLE"
        self.start_time=0.0
        self.path_length=0.0

        self.finish_reached=False

        # potential map
        self.pot_map_on=False
        self.pot_res=(80,60)
        self.pot_surf=None
        # we do a pot_surf if needed later

        # record wind/grad states at "S" time
        self.start_wind_state=False
        self.start_grad_state=False

        # haptic wind mode => "normal" or "inverse"
        self.haptic_wind_mode="normal"

        # for time stepping
        self.last_time=time.time()

        # toggle wind arrow on drone vs corner
        self.wind_display_in_corner = False

        #######################################################################
        # We'll create 10 different environments
        #######################################################################
        self.test_environments = self.create_test_environments(num_envs=self.TOTAL_RUNS)
        self.current_env_index = 0  # which environment (0..9) we are in

        # load environment 0, set default start/finish
        self.load_environment(0)
        self.set_start_finish_for_run(0)  # sets start_area & finish_area
        self.set_default_guidance_for_run(0)  # sets wind_on / gradient_on
        
        sx = 0.5 * (self.start_area[0] + self.start_area[2])
        sy = 0.5 * (self.start_area[1] + self.start_area[3])
        self.drone_pos = np.array([sx, sy], dtype=float)
        
        self.prev_pos=self.drone_pos.copy()

    def create_test_environments(self, num_envs=10):
        """
        Generate random trees for each environment, each with a fixed seed
        so we get reproducible results for each run.
        """
        envs = []
        for i in range(num_envs):
            # each environment has its own seed => reproducible
            seed_local = 1234 + i
            rloc = random.Random(seed_local)
            trees=[]
            for _ in range(self.num_trees):
                x= rloc.uniform(self.xmin+0.1,self.xmax-0.1)
                y= rloc.uniform(self.ymin+0.1,self.ymax-0.1)
                r= rloc.uniform(self.tree_min_size,self.tree_max_size)
                trees.append((x,y,r))
            env = {
                "trees": trees,
                # we keep the bounding box the same,
                # but you could vary it if you want
            }
            envs.append(env)
        return envs

    def load_environment(self, index):
        """
        load environment #index from self.test_environments
        """
        env = self.test_environments[index]
        self.trees = env["trees"]
        print(f"[INFO] Loaded environment #{index+1} with seed {1234+index}, #trees={len(self.trees)}")

    ############################################################################
    ### NEW: define start/finish depending on run number (odd/even)
    ############################################################################
    def set_start_finish_for_run(self, index):
        """
        run_number = index+1 => 1..10
        odd => start bottom-right, end top-left
        even => start bottom-left, end top-right
        """
        run_number = index + 1
        if run_number % 2 == 1:
            # odd => bottom-right -> top-left
            # bottom-right region
            self.start_area = [ self.xmax-0.3, self.ymin,
                                self.xmax,       self.ymin+0.15 ]
            # top-left region
            self.finish_area= [ self.xmin,       self.ymax-0.15,
                                self.xmin+0.3,   self.ymax ]
        else:
            # even => bottom-left -> top-right
            self.start_area = [ self.xmin,       self.ymin,
                                self.xmin+0.3,   self.ymin+0.15 ]
            self.finish_area= [ self.xmax-0.3,   self.ymax-0.15,
                                self.xmax,       self.ymax ]

    ############################################################################
    ### NEW: default guidance for each run
    ############################################################################
    def set_default_guidance_for_run(self, index):
        """
        runs 1,2,9,10 => no guidance => wind=off, gradient=off
        runs 3..8 => guidance => wind=on, gradient=on
        user can still toggle with keys
        """
        run_number = index + 1
        if run_number in [1,2,9,10]:
            self.wind_on = False
            self.gradient_on = False
        else:
            # runs 3..8 => guidance ON
            self.wind_on = True
            self.gradient_on = True
        print(f"[INFO] Set default guidance for run {run_number}: wind_on={self.wind_on}, grad_on={self.gradient_on}")

    def generate_wind_list(self):
        half_range=math.radians(self.dir_range_deg*0.5)
        dir_rad=math.radians(self.init_dir_deg)
        mag=(self.wind_mag_min+self.wind_mag_max)*0.5
        angle=dir_rad
        for i in range(self.num_wind_steps):
            dm=self.rng.gauss(0.0,self.gauss_sigma)
            mag+=dm
            mag=max(self.wind_mag_min,min(self.wind_mag_max,mag))
            da=self.rng.gauss(0.0,self.gauss_sigma)
            angle+=da
            lo= dir_rad-half_range
            hi= dir_rad+half_range
            if angle<lo: angle=lo
            if angle>hi: angle=hi
            self.wind_data.append((mag,angle))
        print("[INFO] wind_data created length=", len(self.wind_data))

    def make_pot_surf(self):
        w2,h2=self.graphics.surface_right.get_size()
        pot_surf=pygame.Surface((w2,h2))
        nx,ny=self.pot_res
        dx=(self.xmax-self.xmin)/(nx-1)
        dy=(self.ymax-self.ymin)/(ny-1)
        pot_vals=np.zeros((ny,nx),dtype=float)
        vmin=1e9
        vmax=-1e9

        for iy in range(ny):
            sy=self.ymin+ iy*dy
            for ix in range(nx):
                sx=self.xmin+ ix*dx
                p=self.compute_potential(sx,sy)
                pot_vals[iy,ix]=p
                if p< vmin: vmin=p
                if p> vmax: vmax=p

        dv=vmax-vmin if vmax>vmin else 1e-9
        cell_w= w2/(nx-1)
        cell_h= h2/(ny-1)
        for iy in range(ny):
            for ix in range(nx):
                val= pot_vals[iy,ix]
                ratio=(val-vmin)/dv
                ratio= max(0, min(1, ratio))
                rcol=int(ratio*255)
                gcol=int((1.0-ratio)*255)
                color=(rcol,gcol,0)
                rx=int(ix*cell_w)
                ry=int((ny-1-iy)*cell_h)
                rr=int(math.ceil(cell_w))
                rh=int(math.ceil(cell_h))
                pygame.draw.rect(pot_surf,color,(rx,ry,rr,rh))

        self.pot_surf=pot_surf

    def compute_potential(self, x,y):
        """
        Summation from trees + border => 1/(dist-sumr)
        """
        pot=0.0
        for (tx,ty,tr) in self.trees:
            dx=x-tx
            dy=y-ty
            dist= math.hypot(dx,dy)
            sumr= tr+self.drone_radius
            eff= dist- sumr if dist>sumr else 0.01
            if eff<0.01:
                eff=0.01
            pot+= self.repulse_const/ eff

        leftdist= x-(self.xmin+self.drone_radius)
        rightdist= (self.xmax-self.drone_radius)- x
        botdist= y-(self.ymin+self.drone_radius)
        topdist= (self.ymax-self.drone_radius)- y
        for d in [leftdist,rightdist,botdist,topdist]:
            if d<0.05 and d>0:
                pot+= self.repulse_const/d
            elif d<=0:
                pot+= self.repulse_const/0.01

        return pot

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
        dt= now - self.last_time
        if dt>0.05:
            dt=0.05
        self.last_time= now

        keyups, mp= self.graphics.get_events()
        for k in keyups:
            if k== ord('q'):
                # quit no save
                sys.exit(0)
            elif k== ord('r'):
                # reset environment
                self.reset_env()
            elif k== ord('w'):
                self.wind_on=not self.wind_on
                if not self.wind_on:
                    self.wind_idx=0
                print("[INFO] wind =>", self.wind_on)
            elif k== ord('g'):
                self.gradient_on= not self.gradient_on
                print("[INFO] gradient =>", self.gradient_on)
            elif k== ord('s'):
                # start only if IDLE
                if self.state=="IDLE":
                    self.start_trial()
            elif k== ord('e'):
                # end + auto save
                if self.state=="RUNNING":
                    self.state="FINISHED"
                self.save_trial()

                # if we haven't done all runs, move on
                if self.current_env_index < self.TOTAL_RUNS - 1:
                    self.current_env_index += 1
                    # load next environment
                    self.reset_env(load_new_env=True)
                    print(f"[INFO] Moved to environment #{self.current_env_index+1}")
                else:
                    print("[INFO] All runs completed!")
            elif k== ord('v'):
                self.pot_map_on= not self.pot_map_on
                if self.pot_map_on:
                    self.make_pot_surf()
                print("[INFO] pot_map =>", self.pot_map_on)
            elif k== ord('z'):
                # toggle haptic wind mode
                if self.haptic_wind_mode=="normal":
                    self.haptic_wind_mode="inverse"
                else:
                    self.haptic_wind_mode="normal"
                print("[INFO] Haptic wind mode =>", self.haptic_wind_mode)
            elif k== ord('d'):
                self.wind_display_in_corner = not self.wind_display_in_corner
                print("[INFO] Wind display in corner =>", self.wind_display_in_corner)

        self.mouse_pos=np.array(mp)

        if self.state=="RUNNING":
            self.update_sim(dt)

        self.graphics.erase_surfaces()

        # compute haptic force
        f_wind= self.get_current_wind() if (self.wind_on and self.state=="RUNNING") else np.array([0,0],dtype=float)
        f_grad= np.array([0,0],dtype=float)
        if self.gradient_on and self.state=="RUNNING":
            f_grad= self.compute_gradient_force()

        if self.haptic_wind_mode=="normal":
            wind_contrib= -f_wind
        else:
            wind_contrib= f_wind
            
        f_att = self.compute_attractive_force()

        hf_x= wind_contrib[0] + f_grad[0] - self.bump_force[0] + f_att[0]
        hf_y= wind_contrib[1] + f_grad[1] - self.bump_force[1] + f_att[1]

        # left-plane rendering
        elapsed_time=0.0
        if self.state in ("RUNNING","FINISHED"):
            end_t= time.time()
            elapsed_time= end_t - self.start_time

        net_fx= getattr(self,"latest_fx",0)
        net_fy= getattr(self,"latest_fy",0)
        self.graphics.render_left(
            self.handle_pos, (net_fx, net_fy),
            self.state, self.wind_on, self.gradient_on,
            elapsed_time, self.haptic_wind_mode
        )

        #######################################################################
        # For the right-plane, pass in a few flags to show start/end/done messages
        #######################################################################
        show_start_msg = (self.state=="IDLE")  # "Press S to start"
        show_end_msg   = (self.finish_reached and self.state=="FINISHED" and self.current_env_index < self.TOTAL_RUNS)
        # If we've done all 10, show "experiment done"
        # We'll consider "done" if current_env_index == TOTAL_RUNS-1 AND we've already finished
        show_done_msg  = (self.finish_reached and self.state=="FINISHED" and (self.current_env_index == self.TOTAL_RUNS-1))

        self.graphics.render_right(
            run_number=self.current_env_index+1,
            total_runs=self.TOTAL_RUNS,
            drone_pos=self.drone_pos,
            drone_radius=self.drone_radius,
            walls=(self.xmin,self.xmax,self.ymin,self.ymax),
            trees=self.trees,
            collision_count=self.collision_count,
            wind_vec=f_wind,
            wind_on=self.wind_on,
            gradient_on=self.gradient_on,
            start_area=self.start_area,
            finish_area=self.finish_area,
            pot_map_on=self.pot_map_on,
            pot_surf=self.graphics.pot_surf,
            wind_in_corner=self.wind_display_in_corner,
            show_start_message=show_start_msg,
            show_end_message=show_end_msg,
            show_done_message=show_done_msg
        )
        self.graphics.finalize()

        # update device or pseudo-haptics
        haptic_force= np.array([hf_x,hf_y],dtype=float)
        if self.physics.is_device_connected():
            self.physics.update_force(haptic_force)
        else:
            newp= self.graphics.sim_forces(self.handle_pos,haptic_force,self.mouse_pos)
            self.handle_pos[:]= newp

    def start_trial(self):
        print("[INFO] Start trial => RUNNING")
        self.state="RUNNING"
        self.start_time= time.time()
        self.path_length=0.0
        self.prev_pos= self.drone_pos.copy()
        self.finish_reached=False
        self.start_wind_state= self.wind_on
        self.start_grad_state= self.gradient_on

    def update_sim(self, dt):
        f_user= self.compute_user_force()
        f_wind= self.get_current_wind() if self.wind_on else np.array([0,0],dtype=float)
        self.update_bump(dt)
        f_grad= np.array([0,0],dtype=float)
        if self.gradient_on:
            f_grad= self.compute_gradient_force()

        if self.haptic_wind_mode=="normal":
            fx= f_user[0] - f_wind[0] + self.bump_force[0] + f_grad[0]
            fy= f_user[1] - f_wind[1] + self.bump_force[1] + f_grad[1]
        else:
            fx= f_user[0] + f_wind[0] + self.bump_force[0] + f_grad[0]
            fy= f_user[1] + f_wind[1] + self.bump_force[1] + f_grad[1]
        self.latest_fx= fx
        self.latest_fy= fy

        ax= fx/self.mass
        ay= fy/self.mass
        self.drone_vel[0]+= ax*dt
        self.drone_vel[1]+= ay*dt
        self.drone_vel*= self.damping
        oldp= self.drone_pos.copy()
        self.drone_pos+= self.drone_vel*dt
        move_dist= np.linalg.norm(self.drone_pos-oldp)
        self.path_length+= move_dist

        # check collisions
        self.check_wall_collision_threshold()
        self.check_tree_collision_threshold()

        # check finish
        if self.is_in_area(self.drone_pos, self.finish_area):
            if not self.finish_reached:
                print("[INFO] Finish reached!")
            self.finish_reached=True
            self.state="FINISHED"
            # no auto-savelogic here; user must press E

    def compute_user_force(self):
        wL,hL= self.graphics.surface_left.get_size()
        cx,cy= wL//2,hL//2
        dx= self.handle_pos[0]- cx
        dy= self.handle_pos[1]- cy
        scale_factor=0.01
        return np.array([dx*scale_factor, -dy*scale_factor],dtype=float)

    def get_current_wind(self):
        (mag,ang)= self.wind_data[self.wind_idx]
        self.wind_idx= (self.wind_idx+1)% len(self.wind_data)
        return np.array([mag*math.cos(ang), mag*math.sin(ang)],dtype=float)
    
    
    def compute_gradient_force(self):
       """
       Linearly increasing repulsive force from trees and walls.
       Force is only applied within a certain distance.
       """
       repulse_radius = 0.1  # meters
       max_force = 1.0       # Newtons
   
       x, y = self.drone_pos
       fx, fy = 0.0, 0.0
   
       # Trees
       for (tx, ty, tr) in self.trees:
           dx = x - tx
           dy = y - ty
           dist = math.hypot(dx, dy)
           sumr = tr + self.drone_radius
           eff_dist = dist - sumr
           if eff_dist < repulse_radius:
               if eff_dist < 0.001:
                   eff_dist = 0.001  # avoid divide-by-zero
               direction = np.array([dx, dy]) / dist
               # Linearly scale the force: 0 N at 0.1 m, max at contact
               force_mag = max_force * (repulse_radius - eff_dist) / repulse_radius
               f = force_mag * direction
               fx += f[0]
               fy += f[1]
   
       # Walls (similar logic)
       # Left wall
       d = x - (self.xmin + self.drone_radius)
       if 0 < d < repulse_radius:
           fx += max_force * (repulse_radius - d) / repulse_radius
       # Right wall
       d = (self.xmax - self.drone_radius) - x
       if 0 < d < repulse_radius:
           fx -= max_force * (repulse_radius - d) / repulse_radius
       # Bottom wall
       d = y - (self.ymin + self.drone_radius)
       if 0 < d < repulse_radius:
           fy += max_force * (repulse_radius - d) / repulse_radius
       # Top wall
       d = (self.ymax - self.drone_radius) - y
       if 0 < d < repulse_radius:
           fy -= max_force * (repulse_radius - d) / repulse_radius
   
       return np.array([fx, fy], dtype=float)
   
    def compute_attractive_force(self):
       """
       Attractive force pulling toward center of finish area.
       Increases linearly with distance, capped at max.
       """
       max_force = 1.5  # Newtons
       attract_radius = 2.6  # max distance to feel attraction
   
       # Center of the goal area
       gx = 0.5 * (self.finish_area[0] + self.finish_area[2])
       gy = 0.5 * (self.finish_area[1] + self.finish_area[3])
       goal_pos = np.array([gx, gy], dtype=float)
   
       delta = goal_pos - self.drone_pos
       dist = np.linalg.norm(delta)
   
       if dist < 1e-6:
           return np.array([0.0, 0.0], dtype=float)
   
       direction = delta / dist
       force_mag = min(max_force, (dist / attract_radius) * max_force)
   
       return force_mag * direction

    def update_bump(self, dt):
        if self.bump_ttl>0:
            self.bump_ttl-= dt
            if self.bump_ttl<=0:
                self.bump_ttl=0
                self.bump_force[:]=0
            else:
                fade= math.exp(-self.bump_decay*dt)
                self.bump_force*= fade

    def check_wall_collision_threshold(self):
        incol= self.is_in_wall_collision()
        if incol:
            if not self.wall_collision:
                self.collision_count+=1
                self.trigger_bump()
                self.wall_collision=True
        else:
            if self.wall_collision:
                distb= self.dist_to_wall()
                if distb> self.leave_threshold:
                    self.wall_collision=False

    def is_in_wall_collision(self):
        col=False
        if self.drone_pos[0]< self.xmin+self.drone_radius:
            self.drone_pos[0]= self.xmin+self.drone_radius
            self.drone_vel[0]*= -0.5
            col=True
        if self.drone_pos[0]> self.xmax-self.drone_radius:
            self.drone_pos[0]= self.xmax-self.drone_radius
            self.drone_vel[0]*= -0.5
            col=True
        if self.drone_pos[1]< self.ymin+self.drone_radius:
            self.drone_pos[1]= self.ymin+self.drone_radius
            self.drone_vel[1]*= -0.5
            col=True
        if self.drone_pos[1]> self.ymax-self.drone_radius:
            self.drone_pos[1]= self.ymax-self.drone_radius
            self.drone_vel[1]*= -0.5
            col=True
        return col

    def dist_to_wall(self):
        dx_left= self.drone_pos[0]- (self.xmin+self.drone_radius)
        dx_right= (self.xmax-self.drone_radius)- self.drone_pos[0]
        dy_bot= self.drone_pos[1]- (self.ymin+self.drone_radius)
        dy_top= (self.ymax-self.drone_radius)- self.drone_pos[1]
        return min(dx_left,dx_right,dy_bot,dy_top)

    def check_tree_collision_threshold(self):
        for i,(tx,ty,tr) in enumerate(self.trees):
            dx= self.drone_pos[0]- tx
            dy= self.drone_pos[1]- ty
            dist= math.hypot(dx,dy)
            sumr= tr+self.drone_radius
            if dist< sumr:
                # collision
                if i not in self.tree_collision_set:
                    overlap= sumr- dist
                    if dist>1e-6:
                        nx= dx/dist
                        ny= dy/dist
                        self.drone_pos[0]+= nx*overlap
                        self.drone_pos[1]+= ny*overlap
                    else:
                        self.drone_pos[0]+=0.0001
                    self.drone_vel*= -0.5
                    self.collision_count+=1
                    self.trigger_bump()
                    self.tree_collision_set.add(i)
            else:
                # check if we've left collision
                if i in self.tree_collision_set:
                    if dist> sumr+ self.leave_threshold:
                        self.tree_collision_set.remove(i)

    def is_in_area(self, pos, area):
        (axmin,aymin,axmax,aymax)=area
        return (pos[0]>=axmin and pos[0]<=axmax and
                pos[1]>=aymin and pos[1]<=aymax)

    def trigger_bump(self):
        mag=0.5
        ang=random.random()*2*math.pi
        self.bump_force=np.array([mag*math.cos(ang), mag*math.sin(ang)])
        self.bump_ttl=0.5

    def save_trial(self):
        end_t= time.time()
        dt=0.0
        if self.state=="RUNNING":
            dt= end_t- self.start_time
            self.state="FINISHED"
        row={
            "RunNum": self.current_env_index+1,
            "Collisions": self.collision_count,
            "TimeSec": round(dt,2),
            "PathLength": round(self.path_length,2),
            "FinishReached": "YES" if self.finish_reached else "NO",
            "WindAtStart": "YES" if self.start_wind_state else "NO",
            "GradAtStart": "YES" if self.start_grad_state else "NO"
        }
        print("[INFO] Saving trial =>", row)
        fname="drone-results.xlsx"
        if not os.path.isfile(fname):
            df= pd.DataFrame([row])
            df.to_excel(fname,index=False)
        else:
            old= pd.read_excel(fname)
            new= pd.concat([old,pd.DataFrame([row])],ignore_index=True)
            new.to_excel(fname,index=False)
        print("[INFO] Results appended to", fname)

    ############################################################################
    # reset_env with optional load_new_env => load the new environment and config
    ############################################################################
    def reset_env(self, load_new_env=False):
        if self.state=="RUNNING":
            # if we do reset in the middle, we discard that trial (no save).
            pass
        self.state="IDLE"

        # re-center the handle
        wL,hL=self.graphics.surface_left.get_size()
        self.handle_pos[:]=[wL//2,hL//2]

        
        
        
        self.drone_vel[:]=0
        self.collision_count=0
        self.wall_collision=False
        self.tree_collision_set.clear()
        self.bump_force[:]=0
        self.bump_ttl=0.0

        self.wind_idx=0
        self.finish_reached=False
        self.path_length=0.0

        if load_new_env:
            # load new environment for self.current_env_index
            self.load_environment(self.current_env_index)
            self.set_start_finish_for_run(self.current_env_index)
            self.set_default_guidance_for_run(self.current_env_index)
            
            

        if self.pot_map_on:
            self.make_pot_surf()
            
        # reset drone
        sx = 0.5 * (self.start_area[0] + self.start_area[2])
        sy = 0.5 * (self.start_area[1] + self.start_area[3])
        self.drone_pos[:] = [sx, sy]

    def end_trial(self):
        pass


if __name__=="__main__":
    app = DroneWithBorderGradient()
    app.run()
