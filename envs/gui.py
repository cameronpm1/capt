import numpy as np
import pyvista as pv
import threading
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from typing import Any, Dict, Type, Optional


from dynamics.base_dynamics import baseDynamics
from dynamics.sat_dynamics import satelliteDynamics
from dynamics.dynamic_object import dynamicObject

def threaded(fn):
    """Call a function using a thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class gui(object):

    '''
    DOES NOT WORK IF self.vista = True
    '''

    def __init__(
            self,
            rate: int = 50,
            xlim: list[float] = [-5,5],
            ylim: list[float] = [-5,5],
            zlim: list[float] = [-5,5],
            vista: bool = False,
            gif: bool = True,
            dim: int = 3,
    ):
        self.rate = rate
        self.xlim = xlim
        self.ylim = zlim
        self.zlim = ylim
        self.vista = vista

        self.azim = -120
        self.elev = 20
        self.incr = 0.5
        self.frames = []
        self.gif = gif
        self.dim = dim


    def call_back(self, misc=None):
        if self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.plot(command)
        return True

    def terminate(self):
        plt.close('all')        

    def __call__(self, pipe):
        print('starting plotter...')
        self._fig = plt.figure(figsize=(10, 10))

        if self.dim == 3:
            self._ax1 = self._fig.add_subplot(1, 1, 1, projection='3d')
        if self.dim == 2:
            self._ax1 = self._fig.add_subplot(1,1,1)
        
        self.pipe = pipe
        timer = self._fig.canvas.new_timer(interval=1)
        timer.add_callback(self.call_back)
        timer.start()
        plt.show()

        print('...done')

    def plot_object2d(self, object1) -> None:
        path_skip = 1 #dont plot the first point in goal path (may have to be tuned)
        self._ax1.clear()
        self._ax1.grid(False)

        points = object1['points']
        lines = object1['lines']

        if 'obstacles' in object1.keys():
            for center in object1['obstacles']:
                self._ax1.scatter(center[0],center[1],color='r')

        self._ax1.scatter(points[0][0],points[0][1],color='k')
        
        
        if 'goal' in object1.keys():
            #plot current goal path
            if object1['goal'] is not None:
                self._ax1.plot(object1['goal'][path_skip:,0],object1['goal'][path_skip:,1],color='g')
        if 'point cloud' in object1.keys():
            #remove point cloud data outside of axis limits
            for i,point in reversed(list(enumerate(object1['point cloud']))):
                if (point[0] < self.xlim[0] or point[0] > self.xlim[1]) or (point[1] < self.ylim[0] or point[1] > self.ylim[1]):
                    object1['point cloud']=np.delete(object1['point cloud'], i, 0) 
            self._ax1.scatter(object1['point cloud'][:][:,0],object1['point cloud'][:][:,1], color='r',s=8)
        
        if 'final goal' in object1.keys():
            self._ax1.scatter(object1['final goal'][0],object1['final goal'][1], color='g', s=40)

        self._ax1.set_xlim(self.xlim[0], self.xlim[1])
        self._ax1.set_xlim(self.ylim[0], self.ylim[1])
      

    def plot_object3d(self, object1) -> None:
        path_skip = 1 #dont plot the first point in goal path (may have to be tuned)
        self._ax1.clear()
        self._ax1.grid(False)
        points = object1['points']
        lines = object1['lines']
        
        #for line in object1['lines']:
            #self._ax1.plot([points[line[0]][0],points[line[1]][0]],
                            #[points[line[0]][1],points[line[1]][1]],
                            #[points[line[0]][2],points[line[1]][2]], color="k")
        if 'obstacles' in object1.keys():
            for center in object1['obstacles']:
                self.plot_sat(self._ax1,scale=0.3,center=center,color='r')

        center = np.average(points, axis=0) + np.array([-0.5,-1,-0.5])
        self.plot_sat(self._ax1,scale=0.5,center=center,color='w')
        
        
        if 'goal' in object1.keys():
            #plot current goal path
            if object1['goal'] is not None:
                self._ax1.plot(object1['goal'][path_skip:,0],object1['goal'][path_skip:,1],object1['goal'][path_skip:,2],color='g')
        if 'point cloud' in object1.keys():
            #remove point cloud data outside of axis limits
            for i,point in reversed(list(enumerate(object1['point cloud']))):
                if (point[0] < self.xlim[0] or point[0] > self.xlim[1]) or (point[1] < self.ylim[0] or point[1] > self.ylim[1]) or (point[2] < self.zlim[0] or point[2] > self.zlim[1]):
                    object1['point cloud']=np.delete(object1['point cloud'], i, 0) 
            self._ax1.scatter(object1['point cloud'][:][:,0],object1['point cloud'][:][:,1],object1['point cloud'][:][:,2], color='r',s=8)
        
        if 'final goal' in object1.keys():
            self._ax1.scatter(object1['final goal'][0],object1['final goal'][1],object1['final goal'][2], color='g', s=40)
        #self._ax1.set_xticks(np.linspace(self.xlim[0],self.xlim[1],10))
        #self._ax1.set_yticks(np.linspace(self.ylim[0],self.ylim[1],10))
        #self._ax1.set_zticks(np.linspace(self.zlim[0],self.zlim[1],10))
            
        

        self._ax1.xaxis.set_pane_color('k')
        self._ax1.yaxis.set_pane_color('k')
        self._ax1.zaxis.set_pane_color('k')
        
        '''
        self.azim += 0.4
        if self.elev > 30:
            self.incr = -0.5
        if self.elev < 10:
            self.incr = 0.4
        self.elev += self.incr
        '''
        
        
        self._ax1.view_init(elev=self.elev, azim=self.azim)
        self.set_axes_equal(self._ax1)
        self._fig.canvas.draw()
        #if self.gif:
        #    image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype='uint8')
        #    print(image)
        #    image = image.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
        #    self.frames.append(image)
            

    def plot(self, objects) -> None:
        self._ax1.clear()
        if self.dim == 3:
            self.plot_object3d(objects)
        if self.dim == 2:
            self.plot_object2d(objects)

        self._fig.canvas.draw()

    def set_axes_equal(self,ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        '''
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        '''

        y_range = 40
        y_middle = np.mean(17.5)
        x_range = 20
        x_middle = np.mean(0)
        z_range = 20
        z_middle = np.mean(0)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_sat(
            self,
            ax,
            scale=1.0,
            center=[0,0,0],
            color='w'
        ):


        wing_len = 3
        wing_hs = 8
        wing_vs = 4
        wing_sep = 0.5

        box_x = np.array([0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0])*scale + center[0]
        box_y = np.array([0,2,2,0,0,0,0,0,0,2,2,2,2,2,2,0])*scale + center[1]
        box_z = np.array([0,0,0,0,0,1,1,0,1,1,0,1,1,0,1,1])*scale + center[2]
        ax.plot(box_x,box_y,box_z,color=color)

        for i in range(wing_hs+1):

            x = np.array([0-wing_sep-wing_len/8*i,0-wing_sep-wing_len/8*i])*scale + center[0]
            y = np.array([0.5,1.5])*scale + center[1]
            z = np.array([0.5,0.5])*scale + center[2]
            ax.plot(x,y,z,color=color)
            x = np.array([1+wing_sep+wing_len/8*i,1+wing_sep+wing_len/8*i])*scale + center[0]
            y = np.array([0.5,1.5])*scale + center[1]
            z = np.array([0.5,0.5])*scale + center[2]
            ax.plot(x,y,z,color=color)

        for i in range(wing_vs+1):
            x = np.array([0-wing_sep,0-wing_sep-wing_len])*scale + center[0]
            y = np.array([0.5+1/wing_vs*i, 0.5+1/wing_vs*i])*scale + center[1]
            z = np.array([0.5,0.5])*scale + center[2]
            ax.plot(x,y,z,color=color)
            x = np.array([1+wing_sep,1+wing_sep+wing_len])*scale + center[0]
            y = np.array([0.5+1/wing_vs*i, 0.5+1/wing_vs*i])*scale + center[1]
            z = np.array([0.5,0.5])*scale + center[2]
            ax.plot(x,y,z,color=color)

        x = np.array([0,-wing_sep])*scale + center[0]
        y = np.array([1,1])*scale + center[1]
        z = np.array([0.5,0.5])*scale + center[2]
        ax.plot(x,y,z,color=color)
        
        x = np.array([1,1+wing_sep])*scale + center[0]
        y = np.array([1,1])*scale + center[1]
        z = np.array([0.5,0.5])*scale + center[2]
        ax.plot(x,y,z,color=color)

    def plot_ball(self,ax,sp=4,s=8,center=[0,0,0],color='red',scale=1):
        angles = np.linspace(-np.pi/2,np.pi/2,sp)
        theta = np.linspace(0,np.pi*2,s)
        x = np.cos(angles)
        y = np.zeros(np.shape(angles))
        z = np.sin(angles)
        past = None
        for t in theta:
            dcm = np.array([[np.cos(t), -np.sin(t), 0],[np.sin(t), np.cos(t), 0], [0, 0, 1]])
            new = np.matmul(dcm,[x,y,z])
            ax.plot(new[0]*scale+center[0],new[1]*scale+center[1],new[2]*scale+center[2],color=color)
            if past is not None:
                for i in range(len(past[0])-2):
                    ax.plot(np.array([past[0][i+1],new[0][i+1]])*scale+center[0],np.array([past[1][i+1],new[1][i+1]])*scale+center[1],np.array([past[2][i+1],new[2][i+1]])*scale+center[2],color=color)
                    if i < len(past[0])-3:
                        ax.plot(np.array([past[0][i+1],new[0][i+2]])*scale+center[0],np.array([past[1][i+1],new[1][i+2]])*scale+center[1],np.array([past[2][i+1],new[2][i+2]])*scale+center[2],color=color)
            past = new

class Renderer:
    """ send data to gui and invoke plotting """

    def __init__(
        self,
        xlim: list[float] = [-5,5],
        ylim: list[float] = [-5,5],
        zlim: list[float] = [-5,5],
        vista: bool = False,
        dim: int = 3,
    ):
        self.vista = vista

        self.dim = dim

        if not self.vista:
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = gui(xlim=xlim,ylim=ylim,zlim=zlim,vista=vista,dim=dim)
            self.plot_process = mp.Process(
                target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        else:
            self.plotter = pv.Plotter()
            self.plotter.show(interactive_update=True)
            self.plotter.set_position([-10,0,10])
            self.plotter.fly_to([0,0,0])


    def plot(self, data):
        if not self.vista:
            send = self.plot_pipe.send
            if data is not None:
                send(data)
            else:
                send(None)
        else:
            self.plotter.clear_actors()
            for o in data:
                actor = self.plotter.add_mesh(o, color='black', style='wireframe', line_width=1)
                
            self.plotter.update()



    