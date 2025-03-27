# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:38:12 2025

@author: marti
"""
import sys
import math
import time
import numpy as np
import random
import pandas as pd
import os
import gc
import pygame

import serial.tools.list_ports
from HaplyHAPI import Board, Device, Mechanisms, Pantograph

class Physics:
    def __init__(self, reverse_motor_order=False, hardware_version=2):
        self.reverse_motor_order = reverse_motor_order
        self.hardware_version = hardware_version
        self.device_present = False

        self.l1=0.07
        self.l2=0.09
        self.d =0.038

        self.port = self.serial_ports()
        if self.port:
            print("Board found on port %s" % self.port[0])
            self.haplyBoard = Board("test", self.port[0], 0)
            self.device = Device(5, self.haplyBoard)
            self.pantograph = Pantograph(self.hardware_version)
            self.device.set_mechanism(self.pantograph)

            if self.hardware_version==2:
                # If you suspect reversed directions, try flipping these or 0->1
                self.device.add_actuator(1,1,2)
                self.device.add_actuator(2,0,1)
                self.device.add_encoder(1,1,241,10752,2)
                self.device.add_encoder(2,0,-61,10752,1)
            elif self.hardware_version==3:
                pass

            self.device.device_set_parameters()

            import numpy as np
            start_time=time.time()
            while True:
                if not self.haplyBoard.data_available():
                    self.device.set_device_torques(np.zeros(2))
                    self.device.device_write_torques()
                    time.sleep(0.001)
                    if time.time()-start_time>5.0:
                        raise ValueError("Haply present, but no data!")
                else:
                    print("[PHYSICS]: Haply found & streaming. Ready!")
                    break
            self.device_present= True
        else:
            print("[PHYSICS]: No device found.")
            self.device_present= False

    def serial_ports(self):
        ports= list(serial.tools.list_ports.comports())
        result=[]
        for p in ports:
            try:
                port= p.device
                s= serial.Serial(port)
                s.close()
                if p.description.startswith("Arduino Zero"):
                    result.append(port)
            except (OSError,serial.SerialException):
                pass
        return result

    def is_device_connected(self):
        return self.device_present

    def get_device_pos(self):
        if not self.device_present:
            raise ValueError("[PHYSICS] get_device_pos() called, no device!")
        self.device.device_read_data()
        angles= self.device.get_device_angles()
        dev_pos= self.device.get_device_position(angles)

        import math
        a1= math.radians(angles[0])
        a2= math.radians(angles[1])
        pA0=(0.0,0.0)
        pB0=(self.d,0.0)
        pA =(self.l1*math.cos(a1), self.l1*math.sin(a1))
        pB =(self.l1*math.cos(a2)+ self.d, self.l1*math.sin(a2))
        return pA0,pB0,pA,pB, dev_pos

    def update_force(self, f_2d):
        if not self.device_present:
            return
        import numpy as np
        # Invert Y
        force_to_send= np.array([-f_2d[0], -f_2d[1]],dtype=float)
        self.device.set_device_torques(force_to_send)
        self.device.device_write_torques()
        time.sleep(0.001)

    def close(self):
        """
        Properly close the Haply device and its serial port, so it can 
        be rediscovered if we start a new experiment in the same Python process.
        """
        if self.device_present:
            # zero torques
            self.device.set_device_torques([0,0])
            self.device.device_write_torques()
            time.sleep(0.1)   # small pause
            # Also close the underlying serial port in self.haplyBoard
            try:
                # The Board object stores the Serial in something like self._Board__port
                # or self.__port.  We'll do:
                self.haplyBoard._Board__port.close() 
            except:
                pass
        print("[PHYSICS] closed.")


def draw_arrow(surface, start, angle, arrow_len=60, color=(0,0,255), width=3):
    end_x= start[0]+ arrow_len*math.cos(angle)
    end_y= start[1]- arrow_len*math.sin(angle)
    pygame.draw.line(surface,color,start,(end_x,end_y),width)
    head_len=10
    head_angle= math.radians(30)
    ldir= angle+ head_angle
    rdir= angle- head_angle
    lx= end_x- head_len*math.cos(ldir)
    ly= end_y+ head_len*math.sin(ldir)
    rx= end_x- head_len*math.cos(rdir)
    ry= end_y+ head_len*math.sin(rdir)
    pygame.draw.line(surface,color,(end_x,end_y),(lx,ly),width)
    pygame.draw.line(surface,color,(end_x,end_y),(rx,ry),width)

class Graphics:
    def __init__(self, device_connected, window_size=(1200,600)):
        pygame.init()
        self.window_size= window_size
        self.window= pygame.display.set_mode(window_size)
        pygame.display.set_caption("Drone with Big Force Debug")

        self.surface_left= pygame.Surface((window_size[0]//2,window_size[1]))
        self.surface_right= pygame.Surface((window_size[0]//2,window_size[1]))

        self.clock= pygame.time.Clock()
        self.FPS=100
        self.font= pygame.font.Font(None,24)
        self.device_connected= device_connected

        self.sim_k=0.4
        self.sim_b=0.8

        wL,hL= self.surface_left.get_size()
        self.handle_min_x=50
        self.handle_max_x=wL-50
        self.handle_min_y=50
        self.handle_max_y=hL-50

        self.drone_scale=300
        self.white=(255,255,255)
        self.lightblue=(200,200,255)
        self.black=(0,0,0)
        self.red=(255,0,0)
        self.green=(0,180,0)

        w2,h2= self.surface_right.get_size()
        self.pot_surf= pygame.Surface((w2,h2))

    def get_events(self):
        ev= pygame.event.get()
        keyups=[]
        for e in ev:
            if e.type== pygame.QUIT:
                sys.exit(0)
            elif e.type== pygame.KEYUP:
                keyups.append(e.key)
        mp= pygame.mouse.get_pos()
        return keyups, mp

    def erase_surfaces(self):
        self.surface_left.fill(self.white)
        self.surface_right.fill(self.lightblue)

    def sim_forces(self, handle_pos, external_force, mouse_pos):
        diff=(mouse_pos[0]- handle_pos[0], mouse_pos[1]- handle_pos[1])
        scaled_f=(external_force[0]*0.05, external_force[1]*0.05)
        k_term=(self.sim_k* diff[0], self.sim_k* diff[1])
        b_term=(scaled_f[0]/ self.sim_b, scaled_f[1]/ self.sim_b)
        dp=(k_term[0]- b_term[0], k_term[1]- b_term[1])
        new_x= handle_pos[0]+ dp[0]
        new_y= handle_pos[1]+ dp[1]
        if new_x< self.handle_min_x: new_x=self.handle_min_x
        if new_x> self.handle_max_x: new_x=self.handle_max_x
        if new_y< self.handle_min_y: new_y=self.handle_min_y
        if new_y> self.handle_max_y: new_y=self.handle_max_y
        return (new_x,new_y)

    def convert_drone_to_screen(self, x_m, y_m):
        w2,h2= self.surface_right.get_size()
        cx,cy= w2//2,h2//2
        sx= cx+ x_m*self.drone_scale
        sy= cy- y_m*self.drone_scale
        return(int(sx),int(sy))

    def render_left(self, handle_pos, total_force, state, wind_on, attraction_on, grad_on, elapsed_time, haptic_wind_mode):
        fm= math.hypot(total_force[0], total_force[1])
        ratio= min(1.0,fm/2.0)
        gb= int(200*(1.0-ratio))
        color=(255,gb,gb)

        rect= pygame.Rect(0,0,40,40)
        rect.center= (int(handle_pos[0]), int(handle_pos[1]))
        pygame.draw.rect(self.surface_left,color, rect, border_radius=6)

        wL,hL= self.surface_left.get_size()
        cx,cy= wL//2,hL//2
        pygame.draw.line(self.surface_left,self.black,(cx-10,cy),(cx+10,cy),2)
        pygame.draw.line(self.surface_left,self.black,(cx,cy-10),(cx,cy+10),2)

        lines=[
            f"STATE: {state}",
            f"Time: {elapsed_time:.2f}s",
            f"Wind: {'ON' if wind_on else 'OFF'}",
            f"Grad: {'ON' if grad_on else 'OFF'}",
            f"Attr: {'ON' if attraction_on else 'OFF'}",
            f"HapticWindMode: {haptic_wind_mode}",
            "",
            "Keys: Q=quit(no save), R=reset(no save)",
            " W=wind, G=gradient, Z=toggle wind mode",
            " S=start, E=end+save, V=pot map",
            " D=toggle wind arrow display",
            "Experiment 1."
        ]
        yoff=10
        for ln in lines:
            txt= self.font.render(ln, True, (0,0,0))
            self.surface_left.blit(txt,(10,yoff))
            yoff+=20
    
    def render_right(self, run_number, total_runs,
                     drone_pos, drone_radius, walls, trees, collision_count,
                     wind_vec, wind_on, attraction_on, gradient_on,
                     start_area, finish_area,
                     pot_map_on, pot_surf,
                     wind_in_corner=False,
                     show_start_message=False,
                     show_end_message=False,
                     show_done_message=False,
                     force_x=0.0,force_y=0.0):
        if pot_map_on:
            self.surface_right.blit(pot_surf,(0,0))
        else:
            self.surface_right.fill(self.lightblue)

        (xmin,xmax,ymin,ymax)= walls
        c_tl= self.convert_drone_to_screen(xmin,ymax)
        c_br= self.convert_drone_to_screen(xmax,ymin)
        rect= pygame.Rect(c_tl[0], c_tl[1], c_br[0]-c_tl[0], c_br[1]- c_tl[1])
        pygame.draw.rect(self.surface_right,(0,0,0),rect,2)

        s_tl= self.convert_drone_to_screen(start_area[0], start_area[3])
        s_br= self.convert_drone_to_screen(start_area[2], start_area[1])
        start_rect= pygame.Rect(s_tl[0], s_tl[1], s_br[0]-s_tl[0], s_br[1]-s_tl[1])
        pygame.draw.rect(self.surface_right,(0,150,0),start_rect,2)

        f_tl= self.convert_drone_to_screen(finish_area[0],finish_area[3])
        f_br= self.convert_drone_to_screen(finish_area[2],finish_area[1])
        finish_rect= pygame.Rect(f_tl[0], f_tl[1], f_br[0]-f_tl[0], f_br[1]-f_tl[1])
        pygame.draw.rect(self.surface_right,(150,0,150), finish_rect,2)

        # trees
        if not pot_map_on:
            for(tx,ty,tr) in trees:
                cc= self.convert_drone_to_screen(tx,ty)
                rp= int(tr*300)
                pygame.draw.circle(self.surface_right,(0,180,0), cc, rp)

        # drone
        cdrone= self.convert_drone_to_screen(drone_pos[0],drone_pos[1])
        rpix= int(drone_radius*300)
        pygame.draw.circle(self.surface_right,(255,0,0), cdrone,rpix)

        t_surf= self.font.render(f"Collisions={collision_count}", True,(0,0,0))
        self.surface_right.blit(t_surf,(10,10))
        force_str = f"Force = [{force_x:.2f}, {force_y:.2f}]"
        force_surf = self.font.render(force_str, True, (0,0,0))
        self.surface_right.blit(force_surf, (10,30))
        w2,h2= self.surface_right.get_size()
        run_text = f"Run {run_number} / {total_runs}"
        run_surf = self.font.render(run_text, True, (0,0,0))
        run_rect = run_surf.get_rect(midtop=(w2//2, 10))
        self.surface_right.blit(run_surf, run_rect)

        if wind_on:
            wmag= math.hypot(wind_vec[0],wind_vec[1])
            if wmag>1e-3:
                angle= math.atan2(wind_vec[1],wind_vec[0])
                if not wind_in_corner:
                    arrow_len= 0.2*wmag
                    end= (cdrone[0] + arrow_len*300*math.cos(angle),
                          cdrone[1] - arrow_len*300*math.sin(angle))
                    pygame.draw.line(self.surface_right,(0,0,255),cdrone,end,3)
                else:
                    corner_pos=(50,50)
                    draw_arrow(self.surface_right, corner_pos, angle,arrow_len=60,color=(0,0,255),width=3)
                    mag_str= f"{wmag:.1f}"
                    txt= self.font.render(f"Wind: {mag_str}", True,(0,0,0))
                    self.surface_right.blit(txt,(corner_pos[0], corner_pos[1]+20))

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

    def finalize(self):
        self.window.blit(self.surface_left,(0,0))
        self.window.blit(self.surface_right,(self.window_size[0]//2,0))
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def show_centered_message(self, msg_lines, sub_msg=""):
        """
        Utility: show large text in the center of the right surface,
        plus an optional sub-message just below it.
        Wait until SPACE or Q pressed.
        Returns 'SPACE' or 'Q' or None if some other key.
        """
        # We'll do a mini loop
        waiting = True
        ret_key = None

        while waiting:
            # handle events
            ev = pygame.event.get()
            for e in ev:
                if e.type==pygame.QUIT:
                    sys.exit(0)
                elif e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_SPACE:
                        ret_key='SPACE'; waiting=False
                    elif e.key==pygame.K_q:
                        ret_key='Q'; waiting=False
                    # ignore others

            # Just erase surfaces
            self.surface_left.fill((255,255,255))
            self.surface_right.fill((200,200,255))

            # draw big text in center of right surface
            w2,h2= self.surface_right.get_size()
            y = h2//2 - 50*len(msg_lines)
            for line in msg_lines:
                txt= self.font.render(line, True, (0,0,0))
                rect= txt.get_rect(midtop=(w2//2, y))
                self.surface_right.blit(txt, rect)
                y += 40
            if sub_msg:
                txt2= self.font.render(sub_msg, True, (0,0,0))
                rect2= txt2.get_rect(midtop=(w2//2, y+40))
                self.surface_right.blit(txt2, rect2)

            # finalize
            self.window.blit(self.surface_left,(0,0))
            self.window.blit(self.surface_right,(self.window_size[0]//2,0))
            pygame.display.flip()
            self.clock.tick(self.FPS)

        return ret_key

###############################################################################
# DRONE with BIG force scale
###############################################################################
class DroneWithBorderGradient:
    def __init__(self):
        self.physics= Physics(reverse_motor_order=False, hardware_version=2)
        self.graphics= Graphics(self.physics.is_device_connected(),(1200,600))
        self.current_study = 1
        self.current_run   = 1
        self.TOTAL_RUNS = 10
        self.start_time = None
        self.end_time = None

        self.user_name = ""
        self.session_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.ask_username()
        
        wL,hL= self.graphics.surface_left.get_size()
        self.handle_pos= np.array([wL//2,hL//2],dtype=float)
        self.mouse_pos= self.handle_pos.copy()

        self.xmin,self.xmax= -1.0,1.0
        self.ymin,self.ymax= -0.8,0.8
        self.start_area = [0,0,0,0]
        self.finish_area= [0,0,0,0]

        self.drone_pos= np.array([self.xmin+0.05,self.ymin+0.05],dtype=float)
        self.drone_vel= np.array([0.0,0.0],dtype=float)
        self.drone_radius= 0.03
        self.mass=1.0
        self.damping=0.96

        self.collision_count=0
        self.wall_collision=False
        self.tree_collision_set=set()
        self.leave_threshold=0.01

        self.num_trees=20
        self.tree_min_size=0.02
        self.tree_max_size=0.10
        self.trees=[]
        
        self.centerAngles = [45, 215, 65, 95, 185, 75, 35, 255, 145, 235]        
        self.all_wind_data = []
        self.num_wind_steps=2000
        self.wind_mag_min=0.4
        self.wind_mag_max=1.0
        self.gauss_sigma=0.05
        self.build_all_wind_data()
        self.wind_on=False
        self.wind_idx=0 

        self.gradient_factor=0.0
        self.attraction_factor=0.0

        self.bump_force= np.array([0.0,0.0],dtype=float)
        self.bump_ttl=0.0
        self.bump_decay=1.5
        self.gradient_on=False
        self.attraction_on=False
        self.repulse_const=0.005

        self.state="IDLE"
        self.start_time=0.0
        self.path_length=0.0
        self.prev_pos=self.drone_pos.copy()
        self.finish_reached=False

        self.pot_map_on=False
        self.pot_res=(80,60)
        self.pot_surf=None
        
        #self.make_pot_surf()

        self.start_wind_state=False
        self.start_grad_state=False
        self.haptic_wind_mode="normal"
        self.last_time=time.time()
        self.wind_display_in_corner=False
        
        self.test_environments = self.create_test_environments(num_envs=self.TOTAL_RUNS)
        self.current_env_index = 0  # which environment (0..9) we are in

        self.special_mode = 0

        # load environment 0, set default start/finish
        self.load_environment(0)
        self.set_start_finish_for_run(0)  # sets start_area & finish_area
        self.set_default_guidance_for_run(0)  # sets wind_on / gradient_on
        
        sx = 0.5 * (self.start_area[0] + self.start_area[2])
        sy = 0.5 * (self.start_area[1] + self.start_area[3])
        self.drone_pos = np.array([sx, sy], dtype=float)
        
        self.show_intro_screens()
        
        # *** Here's the extra scale to make environment forces bigger:
        self.DEBUG_FORCE_SCALE= 5.0  # try 5 or 10

    def ask_username(self):
        """
        Display a simple text input at the start of the experiment to get the user's name.
        Press ENTER to confirm.
        """
        import pygame
        prompt_font= pygame.font.Font(None, 40)
        clock= pygame.time.Clock()

        name_input= ""
        done= False

        while not done:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type== pygame.QUIT:
                    sys.exit(0)
                elif event.type== pygame.KEYDOWN:
                    if event.key== pygame.K_RETURN:
                        done= True
                    elif event.key== pygame.K_BACKSPACE:
                        name_input= name_input[:-1]
                    else:
                        # add character
                        name_input+= event.unicode

            # draw background
            self.graphics.window.fill((255,255,255))
            txt= prompt_font.render("Enter Your Name, then Press ENTER:", True,(0,0,0))
            self.graphics.window.blit(txt,(50,50))

            name_surf= prompt_font.render(name_input, True, (0,0,0))
            self.graphics.window.blit(name_surf,(50,120))

            pygame.display.flip()

        self.user_name= name_input.strip()
        print("[INFO] user_name =>", self.user_name)

    def show_intro_screens(self):
        """
        Optionally show an intro for the entire experiment, or not
        """
        pass
    
    def build_all_wind_data(self):
        """
        Build 10 separate wind_data lists, each using a 90-degree domain.
        For run i => centerAngles[i] +/- 45 degrees => total domain 90 deg.
        We'll do a random walk in magnitude with a small Gaussian in angle,
        but clamp angle to [centerAngle-45, centerAngle+45].
        """
        self.all_wind_data.clear()

        for run_idx in range(10):
            local_list= []
            local_rng= random.Random(777 + run_idx)
            centerA= self.centerAngles[run_idx]
            half_range = 45.0
            mag= 0.5*(self.wind_mag_min + self.wind_mag_max)
            angle= float(centerA)
            for _ in range(self.num_wind_steps):
                dm= local_rng.gauss(0.0, self.gauss_sigma)
                mag+= dm
                if mag< self.wind_mag_min: mag= self.wind_mag_min
                if mag> self.wind_mag_max: mag= self.wind_mag_max
                da= local_rng.gauss(0.0, self.gauss_sigma* 10.0)
                angle+= da
                lowA= centerA - half_range
                highA= centerA + half_range
                if angle< lowA: angle= lowA
                if angle> highA: angle= highA
                local_list.append((mag, math.radians(angle)))
            self.all_wind_data.append(local_list)

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

    def set_default_guidance_for_run(self, index):
        """
        runs 1,2,9,10 => no guidance => wind=off, gradient=off
        runs 3..8 => guidance => wind=on, gradient=on
        user can still toggle with keys
        """
        run_number = index + 1
        if run_number in [1,2,9,10]:
            self.wind_on = True
            self.gradient_on = False
            self.attraction_on = False
        else:
            # runs 3..8 => guidance ON
            self.wind_on = True
            self.gradient_on = False
            self.attraction_on = False
        print(f"[INFO] Set default guidance for run {run_number}: wind_on={self.wind_on}, grad_on={self.gradient_on}, attraction_on={self.attraction_on }")


    def generate_wind_list(self):
        half_range= math.radians(self.dir_range_deg*0.5)
        dir_rad= math.radians(self.init_dir_deg)
        mag=(self.wind_mag_min+self.wind_mag_max)*0.5
        angle= dir_rad
        for i in range(self.num_wind_steps):
            dm= self.rng.gauss(0.0,self.gauss_sigma)
            mag+= dm
            mag= max(self.wind_mag_min,min(self.wind_mag_max,mag))
            da= self.rng.gauss(0.0,self.gauss_sigma)
            angle+= da
            lo= dir_rad-half_range
            hi= dir_rad+ half_range
            if angle<lo: angle=lo
            if angle>hi: angle=hi
            self.wind_data.append((mag,angle))
        print("[INFO] wind_data created length=", len(self.wind_data))

    def make_pot_surf(self):
        """
        We'll build pot_surf as a smooth gradient map around each tree,
        from white inside the tree (center + radius) to black at radius+0.1,
        and black for the rest of the plane.
        """
        w2,h2= self.graphics.surface_right.get_size()

        # let's use a higher resolution for less pixelation
        nx,ny= (300,200)
        dx= (self.xmax- self.xmin)/(nx-1)
        dy= (self.ymax- self.ymin)/(ny-1)

        # create a blank surface
        pot_surf= pygame.Surface((w2,h2))
        pot_surf.fill((0,0,0))

        cell_w= w2/(nx-1)
        cell_h= h2/(ny-1)

        # for each cell, find "best color" among all trees
        # if dist < radius => color= white(255)
        # if radius <= dist < radius+0.1 => fade from 255 => 0
        # else => black => 0
        # if multiple trees => pick the max color => whiter
        for iy in range(ny):
            wy= self.ymin+ iy* dy
            for ix in range(nx):
                wx= self.xmin+ ix* dx

                # figure out how white this pixel is, among all trees
                pixel_color= 0  # default black
                for (tx,ty,tr) in self.trees:
                    dist= math.hypot(wx- tx, wy- ty)

                    if dist< tr:
                        # fully inside => white
                        color_val= 255
                    elif tr <= dist< tr+ 0.1:
                        # fade => 255 => 0
                        # dist= tr => 255, dist= tr+0.1 => 0
                        ratio= 1.0- ((dist- tr)/ 0.1)
                        color_val= int(255* ratio)
                    else:
                        color_val= 0

                    if color_val> pixel_color:
                        pixel_color= color_val

                # draw the cell => grayscale
                c= (pixel_color, pixel_color, pixel_color)
                rx= int(ix* cell_w)
                ry= int((ny-1- iy)* cell_h)
                rr= int(math.ceil(cell_w))
                rh= int(math.ceil(cell_h))
                pygame.draw.rect(pot_surf, c, (rx, ry, rr, rh))

        self.pot_surf= pot_surf
                        
    def compute_potential(self, x,y):
        pot=0.0
        for(tx,ty,tr) in self.trees:
            dx= x- tx
            dy= y- ty
            dist= math.hypot(dx,dy)
            sumr= tr+ self.drone_radius
            if dist> sumr:
                eff= dist- sumr
                if eff<0.01: eff=0.01
                pot+= self.repulse_const/ eff
            else:
                pot+= self.repulse_const/ 0.01

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
            pot+= self.repulse_const/0.01

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

    def main_loop(self):
        """
        We'll do study=1 => 10 runs => then show "Press SPACE => part2"
        study=2 => 10 runs => "SPACE => part3"
        study=3 => 10 runs => "Press Q => done"
        """
        while True:
            # If we've done all 3 studies => done
            if self.current_study>3:
                # final message => "Experiment done, press Q to quit"
                msg_lines = ["All 3 studies done.", "Experiment done.", "Press 'Q' to quit."]
                key = self.graphics.show_centered_message(msg_lines)
                if key=='Q':
                    sys.exit(0)
                # else keep waiting
                continue

            # If the user hasn't done all runs in the current study:
            if self.current_run <= self.RUNS_PER_STUDY:
                self.loop_once()  # update
            else:
                # The study is finished, show the "Press SPACE" screen (except for study 3 => "press Q to quit"?)
                if self.current_study<3:
                    partX = self.current_study+1
                    txt = f"Press 'SPACE' to continue to part {partX} of the experiment."
                    keys= self.graphics.show_centered_message([f"Study {self.current_study} done.",txt], sub_msg="")
                    if keys=='SPACE':
                        self.current_study +=1
                        self.current_run    =1
                        self.reset_study()
                    elif keys=='Q':
                        sys.exit(0)
                else:
                    # study=3 done => "Experiment done, press Q to quit"
                    msg_lines=["Study 3 done.","Experiment done.","Press 'Q' to quit."]
                    keys2= self.graphics.show_centered_message(msg_lines)
                    if keys2=='Q':
                        sys.exit(0)

    def loop_once(self):
        now= time.time()
        dt= now- self.last_time
        if dt>0.05:
            dt=0.05
        self.last_time= now

        keyups, mp= self.graphics.get_events()
        
        for k in keyups:
            if k==ord('q'):
                sys.exit(0)
            elif k==ord('r'):
                self.reset_env()
            elif k==ord('w'):
                self.wind_on= not self.wind_on
                if not self.wind_on:
                    self.wind_idx=0
                print("[INFO] wind =>", self.wind_on)
            elif k==ord('g'):
                self.gradient_on= not self.gradient_on
                self.attraction_on= not self.attraction_on
                print("[INFO] gradient =>", self.gradient_on)
                print("[INFO] attraction =>", self.attraction_on)
            elif k==ord('s'):
                if self.state=="IDLE":
                    self.start_trial()
            elif k== ord('e'):
                if self.state=="RUNNING":
                    self.state="FINISHED"
                self.save_trial()

                # ### We check whether we just finished run#2 or run#8
                # run #2 => index=1 => special_mode=1 => "the next runs have haptic guidance"
                # run #8 => index=7 => special_mode=2 => "the next runs have NO guidance"
                if self.current_env_index< self.TOTAL_RUNS -1:
                    self.current_env_index+=1
                    self.reset_env(load_new_env=True)
                    print(f"[INFO] Moved to environment #{self.current_env_index+1}")
                else:
                    print("[INFO] All runs completed!")
            elif k==ord('v'):
                self.pot_map_on= not self.pot_map_on
                if self.pot_map_on:
                    self.make_pot_surf()
                print("[INFO] pot_map =>", self.pot_map_on)
            elif k==ord('z'):
                if self.haptic_wind_mode=="normal":
                    self.haptic_wind_mode="inverse"
                else:
                    self.haptic_wind_mode="normal"
                print("[INFO] haptic wind =>",self.haptic_wind_mode)
            elif k==ord('d'):
                self.wind_display_in_corner= not self.wind_display_in_corner
                print("[INFO] wind corner =>", self.wind_display_in_corner)

        self.mouse_pos= np.array(mp)

        if self.state=="RUNNING":
            self.update_sim(dt)
        
        f_wind= self.get_current_wind() if(self.wind_on and self.state=="RUNNING") else np.array([0,0],dtype=float)
        f_grad= np.array([0,0],dtype=float)
        f_attraction= np.array([0,0],dtype=float)
        if self.gradient_on and self.state=="RUNNING":
            f_grad= self.compute_gradient_force()
        if self.attraction_on and self.state=="RUNNING":
            f_attraction= self.compute_attractive_force()
        # invert wind if normal, else keep sign
        if self.haptic_wind_mode=="normal":
            wind_contrib= -f_wind
        else:
            wind_contrib= f_wind
        
        hf_x= wind_contrib[0] + f_grad[0] - self.bump_force[0] + f_attraction[0]
        hf_y= wind_contrib[1] + f_grad[1] - self.bump_force[1] + f_attraction[1]

        # *** Multiply environment forces to help feel it
        hf_x*= self.DEBUG_FORCE_SCALE
        hf_y*= self.DEBUG_FORCE_SCALE

        self.graphics.erase_surfaces()

        elapsed_time=0.0
        if self.state in("RUNNING","FINISHED"):
            end_t= time.time()
            elapsed_time= end_t- self.start_time

        net_fx= getattr(self,"latest_fx",0)
        net_fy= getattr(self,"latest_fy",0)

        self.graphics.render_left(
            self.handle_pos, (net_fx, net_fy),
            self.state, self.wind_on, self.gradient_on, self.attraction_on,
            elapsed_time, self.haptic_wind_mode
        )
        
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
            attraction_on=self.attraction_on,
            start_area=self.start_area,
            finish_area=self.finish_area,
            pot_map_on=self.pot_map_on,
            pot_surf=self.graphics.pot_surf,
            wind_in_corner=self.wind_display_in_corner,
            show_start_message=show_start_msg,
            show_end_message=show_end_msg,
            show_done_message=show_done_msg,
            force_x=net_fx,
            force_y=net_fy
        )

        self.graphics.finalize()

        # pass force
        haptic_force= np.array([hf_x,hf_y],dtype=float)

        if self.physics.is_device_connected():
            self.physics.update_force(haptic_force)
            try:
                # read device => place handle
                pA0,pB0,pA,pB,dev_pos= self.physics.get_device_pos()
                scl= 3000.0
                x_screen= 301 + (-dev_pos[0]* scl)
                y_screen= 70  + ( dev_pos[1]* scl)
                self.handle_pos= np.array([x_screen,y_screen],dtype=float)
            except:
                pass
        else:
            newp= self.graphics.sim_forces(self.handle_pos,haptic_force,self.mouse_pos)
            self.handle_pos[:]= newp

    def draw_special_mode_screen(self):
        self.graphics.erase_surfaces()
        surfR= self.graphics.surface_right
        font= self.graphics.font

        w2,h2= surfR.get_size()
        if self.special_mode==1:
            txt1= "The next runs will have haptic guidance."
            txt2= "Attractive force to endpoint, and repulsive force on trees."
            txt3= "Press 'E' to simulate next environment."
            s1= font.render(txt1,True,(0,0,0))
            s2= font.render(txt2,True,(0,0,0))
            s3= font.render(txt3,True,(0,0,0))
            r1= s1.get_rect(center=(w2//2, h2//2 -40))
            r2= s2.get_rect(center=(w2//2, h2//2 ))
            r3= s3.get_rect(center=(w2//2, h2//2 +40))
            surfR.blit(s1,r1)
            surfR.blit(s2,r2)
            surfR.blit(s3,r3)

        elif self.special_mode==2:
            txt1= "For the following runs, there will be no haptic guidance."
            txt2= "Press 'E' to simulate next environment."
            s1= font.render(txt1,True,(0,0,0))
            s2= font.render(txt2,True,(0,0,0))
            r1= s1.get_rect(center=(w2//2, h2//2 - 20))
            r2= s2.get_rect(center=(w2//2, h2//2 + 20))
            surfR.blit(s1,r1)
            surfR.blit(s2,r2)
        
        self.graphics.window.blit(self.graphics.surface_left,(0,0))
        self.graphics.window.blit(self.graphics.surface_right,(self.graphics.window_size[0]//2,0))
        pygame.display.flip()

    def start_trial(self):
        print("[INFO] Start trial => RUNNING")
        self.state="RUNNING"
        self.start_time= time.time()
        self.end_time = None
        self.path_length=0.0
        self.prev_pos= self.drone_pos.copy()
        self.finish_reached=False
        self.start_wind_state= self.wind_on
        self.start_grad_state= self.gradient_on
        self.start_attraction_state= self.attraction_on

    def update_sim(self, dt):
        f_user= self.compute_user_force()
        f_wind= self.get_current_wind() if self.wind_on else np.array([0,0],dtype=float)
        self.update_bump(dt)
        f_grad= np.array([0,0],dtype=float)
        f_attraction= np.array([0,0],dtype=float)
        if self.gradient_on:
            f_grad= self.compute_gradient_force()
        if self.attraction_on:
            f_attraction = self.compute_attractive_force()
        if self.haptic_wind_mode=="normal":
            fx= f_user[0] + -f_wind[0] + self.bump_force[0] + f_grad[0] + f_attraction[0]
            fy= f_user[1] + -f_wind[1] + self.bump_force[1] + f_grad[1] + f_attraction[1]
        else:
            fx= f_user[0] + f_wind[0] + self.bump_force[0] + f_grad[0] + f_attraction[0]
            fy= f_user[1] + f_wind[1] + self.bump_force[1] + f_grad[1] + f_attraction[1]
        self.latest_fx= fx
        self.latest_fy= fy

        ax= fx/self.mass
        ay= fy/self.mass
        self.drone_vel[0]+= ax*dt
        self.drone_vel[1]+= ay*dt
        self.drone_vel*= self.damping
        oldp= self.drone_pos.copy()
        self.drone_pos+= self.drone_vel*dt
        move_dist= np.linalg.norm(self.drone_pos- oldp)
        self.path_length+= move_dist

        self.check_wall_collision_threshold()
        self.check_tree_collision_threshold()

        if self.is_in_area(self.drone_pos, self.finish_area):
            if not self.finish_reached:
                print("[INFO] Finish reached!")
            self.finish_reached = True
            self.end_time = time.time()
            self.state = "FINISHED"

    def compute_user_force(self):
        wL,hL= self.graphics.surface_left.get_size()
        cx,cy= wL//2,hL//2
        dx= self.handle_pos[0]- cx
        dy= self.handle_pos[1]- cy
        scale_factor=0.01
        return np.array([dx*scale_factor, -dy*scale_factor],dtype=float)

    def get_current_wind(self):
        run_idx= self.current_env_index
        wind_list= self.all_wind_data[run_idx]

        (mag, ang)= wind_list[self.wind_idx]
        self.wind_idx= (self.wind_idx+1) % len(wind_list)
        return np.array([mag*math.cos(ang), mag*math.sin(ang)],dtype=float)

    def compute_gradient_force(self):
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
       
       # Vector from drone to goal
       delta = goal_pos - self.drone_pos
       dist = np.linalg.norm(delta)
       
       # If we're basically at the goal, no force
       if dist < 1e-6:
           return np.array([0.0, 0.0], dtype=float)
       
       direction = delta / dist  # unit vector
       # Force grows linearly with distance, but capped at max_force
       force_mag = min(max_force, (dist / attract_radius) * max_force)
        
       # Build the final array
       f_ax = force_mag * direction[0]
       f_ay = force_mag * direction[1]
       return np.array([f_ax, f_ay], dtype=float)

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
            sumr= tr+ self.drone_radius
            if dist< sumr:
                if i not in self.tree_collision_set:
                    overlap= sumr-dist
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
        (axmin,aymin,axmax,aymax)= area
        return (pos[0]>=axmin and pos[0]<=axmax and
                pos[1]>=aymin and pos[1]<=aymax)

    def trigger_bump(self):
        mag=0.5
        ang= random.random()*2*math.pi
        self.bump_force= np.array([mag*math.cos(ang), mag*math.sin(ang)])
        self.bump_ttl=0.5


    def save_trial(self):
        if self.end_time is None and self.state == "RUNNING":
            self.end_time = time.time()
        dt = 0.0
        if self.start_time is not None and self.end_time is not None:
            dt = self.end_time - self.start_time
        mm = int(dt // 60)
        ss = int(dt % 60)
        time_str = f"{mm}:{ss:02d}"
        row={
            "RunNum": self.current_env_index+1,
            "Collisions": self.collision_count,
            "TimeSec": time_str,
            "PathLength": round(self.path_length,2),
            "FinishReached": "YES" if self.finish_reached else "NO",
            "WindAtStart": "YES" if self.start_wind_state else "NO",
            "GradAtStart": "YES" if self.start_grad_state else "NO",
            "UserName": self.user_name,
            "SessionTime": self.session_time_str,
            "Experiment No" : 1
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

    def reset_env(self, load_new_env=False):
        if self.state=="RUNNING":
            pass
        self.state="IDLE"
        wL,hL= self.graphics.surface_left.get_size()
        self.handle_pos[:]= [wL//2,hL//2]
        self.drone_pos[:]= [self.xmin+0.05,self.ymin+0.05]
        self.drone_vel[:]= 0
        self.collision_count=0
        self.wall_collision=False
        self.tree_collision_set.clear()
        self.bump_force[:]=0
        self.bump_ttl=0.0
        self.wind_on=False
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
        
        sx = 0.5 * (self.start_area[0] + self.start_area[2])
        sy = 0.5 * (self.start_area[1] + self.start_area[3])
        self.drone_pos[:] = [sx, sy]
        
    def end_trial(self):
        pass

def run_experiment():
    """
    Creates a DroneWithBorderGradient instance, runs the experiment, 
    then returns once the user is done (press Q).
    """
    app = DroneWithBorderGradient()
    app.run()      # blocks until user is finished
    # At this point, user pressed Q or finished the 10 runs
    # We return so the caller can clean up references
    del app        # delete reference so Python can free
    gc.collect()   # force garbage collection
    print("[INFO] Freed up the old DroneWithBorderGradient object.")

if __name__=="__main__":
    app= DroneWithBorderGradient()
    app.run()
