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
    def __init__(self, hardware_version=2):
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
        # we'll fill or draw surface_right in the final step

    def sim_forces(self, handle_pos, external_force, mouse_pos):
        """
        Pseudo-haptics, clamp handle in 4 directions
        """
        diff = (mouse_pos[0]-handle_pos[0], mouse_pos[1]-handle_pos[1])
        scaled_f = (external_force[0]*0.05, external_force[1]*0.05)
        k_term = (self.sim_k*diff[0], self.sim_k*diff[1])
        b_term = (scaled_f[0]/self.sim_b, scaled_f[1]/self.sim_b)
        dp= (k_term[0]- b_term[0], k_term[1]- b_term[1])

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

    def render_left(self, handle_pos, total_force, state, wind_on, grad_on, elapsed_time, haptic_wind_mode):
        """
        color => based on total_force magnitude
        also display state, time, wind, gradient, haptic wind mode
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
            " S=start, E=end+save, V=pot map"
        ]
        yoff=10
        for ln in lines:
            srf=self.font.render(ln, True,(0,0,0))
            self.surface_left.blit(srf,(10,yoff))
            yoff+=20

    def render_right(self, drone_pos, drone_radius, walls, trees, collision_count,
                     wind_vec, wind_on, gradient_on,
                     start_area, finish_area,
                     pot_map_on, pot_surf):
        if pot_map_on:
            self.surface_right.blit(pot_surf,(0,0))
        else:
            self.surface_right.fill(self.lightblue)

        (xmin,xmax,ymin,ymax)= walls
        c_tl=self.convert_drone_to_screen(xmin,ymax)
        c_br=self.convert_drone_to_screen(xmax,ymin)
        rect=pygame.Rect(c_tl[0], c_tl[1], c_br[0]-c_tl[0], c_br[1]-c_tl[1])
        pygame.draw.rect(self.surface_right,(0,0,0), rect,2)

        # start
        s_tl=self.convert_drone_to_screen(start_area[0], start_area[3])
        s_br=self.convert_drone_to_screen(start_area[2], start_area[1])
        start_rect= pygame.Rect(s_tl[0], s_tl[1], s_br[0]-s_tl[0], s_br[1]-s_tl[1])
        pygame.draw.rect(self.surface_right,(0,150,0),start_rect,2)

        # finish
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

        # wind arrow
        if wind_on:
            wmag=math.hypot(wind_vec[0], wind_vec[1])
            if wmag>1e-3:
                angle=math.atan2(wind_vec[1], wind_vec[0])
                arrow_len=0.2*wmag
                end=(cdrone[0]+ arrow_len*300*math.cos(angle),
                     cdrone[1]- arrow_len*300*math.sin(angle))
                pygame.draw.line(self.surface_right,(0,0,255),cdrone,end,3)

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

        # handle
        wL,hL=self.graphics.surface_left.get_size()
        self.handle_pos=np.array([wL//2,hL//2],dtype=float)
        self.mouse_pos=self.handle_pos.copy()

        # environment
        self.xmin,self.xmax= -1.0,1.0
        self.ymin,self.ymax= -0.8,0.8
        self.start_area=[ self.xmin, self.ymin, self.xmin+0.3, self.ymin+0.15 ]
        self.finish_area=[ self.xmax-0.3, self.ymax-0.15, self.xmax, self.ymax ]

        # drone
        self.drone_pos=np.array([self.xmin+0.05,self.ymin+0.05],dtype=float)
        self.drone_vel=np.array([0.0,0.0],dtype=float)
        self.drone_radius=0.03
        self.mass=1.0
        self.damping=0.96

        # collisions
        self.collision_count=0
        self.wall_collision=False
        self.tree_collision_set=set()
        self.leave_threshold=0.01

        # trees
        self.num_trees=15
        self.tree_min_size=0.02
        self.tree_max_size=0.07
        self.trees=[]
        self.init_trees()

        # wind
        self.wind_on=False
        self.rng=random.Random(3333)
        self.num_wind_steps=2000
        self.wind_data=[]
        self.wind_idx=0
        self.init_dir_deg=30.0
        self.dir_range_deg=90.0
        self.wind_mag_min=0.0
        self.wind_mag_max=1.0  # bigger wind
        self.gauss_sigma=0.05
        self.generate_wind_list()

        # collision bump
        self.bump_force=np.array([0.0,0.0],dtype=float)
        self.bump_ttl=0.0
        self.bump_decay=1.5

        # gradient
        self.gradient_on=False
        self.repulse_const=0.005  # less intense
        # states
        self.state="IDLE"
        self.start_time=0.0
        self.path_length=0.0
        self.prev_pos=self.drone_pos.copy()
        self.finish_reached=False

        # potential map
        self.pot_map_on=False
        self.pot_res=(80,60)
        self.pot_surf=None
        self.make_pot_surf()

        # want to store wind_on/gradient_on at the moment "s" was pressed
        self.start_wind_state=False
        self.start_grad_state=False

        # haptic wind mode => "normal" or "inverse"
        # normal => user feels negative of wind
        # inverse => user feels positive of wind
        self.haptic_wind_mode="normal"

        self.last_time=time.time()

    def init_trees(self):
        random.seed(1234)
        for i in range(self.num_trees):
            x= random.uniform(self.xmin+0.1,self.xmax-0.1)
            y= random.uniform(self.ymin+0.1,self.ymax-0.1)
            r= random.uniform(self.tree_min_size,self.tree_max_size)
            self.trees.append((x,y,r))

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
                if ratio<0: ratio=0
                if ratio>1: ratio=1
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
        sum repulse from trees + border => 1/dist
        """
        pot=0.0
        # trees
        for (tx,ty,tr) in self.trees:
            dx=x-tx
            dy=y-ty
            dist= math.hypot(dx,dy)
            sumr= tr+self.drone_radius
            eff= dist- sumr
            if eff<0.01: eff=0.01
            if dist> sumr:
                pot+= self.repulse_const/ eff
            else:
                pot+= self.repulse_const/ 0.01
        # border
        leftdist= x-(self.xmin+self.drone_radius)
        if leftdist<0.05 and leftdist>0:
            pot+= self.repulse_const/ leftdist
        elif leftdist<=0:
            pot+= self.repulse_const/0.01
        rightdist= (self.xmax-self.drone_radius)- x
        if rightdist<0.05 and rightdist>0:
            pot+= self.repulse_const/ rightdist
        elif rightdist<=0:
            pot+= self.repulse_const/0.01
        botdist= y-(self.ymin+self.drone_radius)
        if botdist<0.05 and botdist>0:
            pot+= self.repulse_const/ botdist
        elif botdist<=0:
            pot+= self.repulse_const/ 0.01
        topdist= (self.ymax-self.drone_radius)- y
        if topdist<0.05 and topdist>0:
            pot+= self.repulse_const/ topdist
        elif topdist<=0:
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
        dt= now- self.last_time
        if dt>0.05:
            dt=0.05
        self.last_time= now

        keyups, mp= self.graphics.get_events()
        for k in keyups:
            if k== ord('q'):
                # quit no save
                sys.exit(0)
            elif k== ord('r'):
                # reset no save
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
                if self.state=="IDLE":
                    self.start_trial()
            elif k== ord('e'):
                # end+auto save
                if self.state=="RUNNING":
                    self.state="FINISHED"
                self.save_trial()
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
                print("[INFO] Haptic wind mode =>",self.haptic_wind_mode)

        self.mouse_pos=np.array(mp)

        if self.state=="RUNNING":
            self.update_sim(dt)

        # build haptic
        f_wind= self.get_current_wind() if (self.wind_on and self.state=="RUNNING") else np.array([0,0],dtype=float)
        # collision => negative
        # gradient => if on => same direction as drone's gradient
        f_grad= np.array([0,0],dtype=float)
        if self.gradient_on and self.state=="RUNNING":
            f_grad= self.compute_gradient_force()
        # bump => negative
        # haptic wind => if normal => -f_wind, if inverse => +f_wind
        if self.haptic_wind_mode=="normal":
            wind_contrib= -f_wind
        else:
            wind_contrib= f_wind

        hf_x= wind_contrib[0] + f_grad[0] - self.bump_force[0]
        hf_y= wind_contrib[1] + f_grad[1] - self.bump_force[1]

        self.graphics.erase_surfaces()

        elapsed_time=0.0
        if self.state in ("RUNNING","FINISHED"):
            # measure time from start_time to now
            # if FINISHED, we don't keep updating but let's do for the display
            end_t= time.time()
            elapsed_time= end_t- self.start_time

        net_fx= getattr(self,"latest_fx",0)
        net_fy= getattr(self,"latest_fy",0)
        self.graphics.render_left(
            self.handle_pos, (net_fx, net_fy),
            self.state, self.wind_on, self.gradient_on,
            elapsed_time, self.haptic_wind_mode
        )
        self.graphics.render_right(
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
            pot_surf=self.graphics.pot_surf
        )
        self.graphics.finalize()

        # update device or pseudo
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
        # record wind/grad states at start
        self.start_wind_state= self.wind_on
        self.start_grad_state= self.gradient_on

    def update_sim(self, dt):
        f_user= self.compute_user_force()
        f_wind= self.get_current_wind() if self.wind_on else np.array([0,0],dtype=float)
        self.update_bump(dt)
        f_grad= np.array([0,0],dtype=float)
        if self.gradient_on:
            f_grad= self.compute_gradient_force()

        fx= f_user[0]+ f_wind[0]+ self.bump_force[0]+ f_grad[0]
        fy= f_user[1]+ f_wind[1]+ self.bump_force[1]+ f_grad[1]
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

        # collisions
        self.check_wall_collision_threshold()
        self.check_tree_collision_threshold()

        # check finish
        if self.is_in_area(self.drone_pos, self.finish_area):
            print("[INFO] Finish reached!")
            self.finish_reached=True
            self.state="FINISHED"
            self.save_trial()

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
        sum of repulse from trees + border => 1/(dist-sumr)^2 or if dist< sumr => collision logic
        plus smaller repulse_const
        """
        x,y= self.drone_pos
        fx,fy= 0.0,0.0
        for (tx,ty,tr) in self.trees:
            dx= x- tx
            dy= y- ty
            dist= math.hypot(dx,dy)
            sumr= tr+self.drone_radius
            if dist> sumr:
                eff= dist- sumr
                if eff<0.01: eff=0.01
                nx= dx/dist
                ny= dy/dist
                val= self.repulse_const/(eff**2)
                fx+= val*nx
                fy+= val*ny
        # border
        leftdist= x-(self.xmin+self.drone_radius)
        if leftdist<0.05 and leftdist>0:
            fx+= self.repulse_const/(leftdist**2)
        rightdist= (self.xmax-self.drone_radius)- x
        if rightdist<0.05 and rightdist>0:
            fx-= self.repulse_const/(rightdist**2)
        botdist= y-(self.ymin+self.drone_radius)
        if botdist<0.05 and botdist>0:
            fy+= self.repulse_const/(botdist**2)
        topdist= (self.ymax-self.drone_radius)- y
        if topdist<0.05 and topdist>0:
            fy-= self.repulse_const/(topdist**2)

        return np.array([fx,fy],dtype=float)

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

        dt = end_t - self.start_time if self.state in ("RUNNING", "FINISHED") else 0.0
    
        if self.state=="RUNNING":
            self.state="FINISHED"

        row={
            "Collisions": self.collision_count,
            "TimeSec": round(dt,2),
            "PathLength": round(self.path_length,2),
            "FinishReached": "YES" if self.finish_reached else "NO",
            # also store wind_on & gradient_on as of when s was pressed
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

    def reset_env(self):
        if self.state=="RUNNING":
            # just discard, no saving
            pass
        self.state="IDLE"
        wL,hL=self.graphics.surface_left.get_size()
        self.handle_pos[:]=[wL//2,hL//2]
        self.drone_pos[:]=[self.xmin+0.05,self.ymin+0.05]
        self.drone_vel[:]=0
        self.collision_count=0
        self.wall_collision=False
        self.tree_collision_set.clear()
        self.bump_force[:]=0
        self.bump_ttl=0.0
        self.wind_on=False
        self.wind_idx=0
        self.finish_reached=False
        self.path_length=0.0

    def end_trial(self):
        # not used, we do save_trial directly
        pass

if __name__=="__main__":
    app= DroneWithBorderGradient()
    app.run()
