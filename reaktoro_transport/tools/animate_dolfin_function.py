from . import *
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation
#import matplotlib.pyplot as plt

def animate_dolfin_function(mesh_2d, func_list, fig, ax_list, level, plot_colorbar, tb_frames):
    # This function animates dolfin functions

    # Inputs:
    # mesh_2d:            dolfin generated mesh
    # func_list:          A list of dolfin functions
    # fig:                Function handle generated by matplotlib.pyplot.subplots
    # ax_list:            A list of ax generated by matplotlib.pyplot
    # level:              Interested contour levels. For example: np.linspace(0,1,11)
    # plot_colorbar:      True or False
    # tb_frames:          Time between frames. Unit: milliseconds

    # Outputs:
    # ani:                FuncAnimation handle

    mesh_x = mesh_2d.coordinates()[:,0]
    mesh_y = mesh_2d.coordinates()[:,1]
    connectivity = mesh_2d.cells()

    triang = Triangulation(mesh_x, mesh_y, connectivity)

    CG_space = FunctionSpace(mesh_2d, 'CG', 1)
    v_to_d = vertex_to_dof_map(CG_space)


    def init():
        for i, ax in enumerate(ax_list):
            cb = ax.tricontourf(triang, func_list[i][0].vector()[v_to_d], levels=level)

        if plot_colorbar==True:
            fig.colorbar(cb)
        #ax.set_ylim(0,1)
        #ax.set_xlim(0,1)
        return cb,

    def update(t):
        for i, ax in enumerate(ax_list):
            ax.set_title('timesteps = ' + str(t))
            cb = ax.tricontourf(triang, func_list[i][t].vector()[v_to_d], levels=level)
            #ln.set_data(x_space, adv_diff_reac_transient_sol_fracture(Pe, Da, epsilon, 0, x_space, t))
        return cb,

    ani = FuncAnimation(fig, update, frames=np.arange(1,len(func_list[0]),1), init_func=init\
                        , blit=True, interval=tb_frames, cache_frame_data=False)

    return ani
