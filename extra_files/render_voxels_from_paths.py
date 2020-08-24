import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import sampler
import cv2
import sys
import numpy as np
import pickle
from binvox_rw import read_as_3d_array
from matplotlib.colors import LightSource
from matplotlib import cm
# Functions from https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib

def cuboid_data(pos, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0),ax=None):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos )
        #light = LightSource(180, 45)
        #illuminated_surface = light.shade(Z, cmap=cm.coolwarm)
        ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, alpha=1, color='blue')
        #        antialiased=False, facecolors=illuminated_surface)

def plotMatrix(ax, matrix):
    # plot a Matrix 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i,j,k] == 1:
                    # to have the 
                    plotCubeAt(pos=(i-0.5,j-0.5,k-0.5), ax=ax) 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
def plot_mesh(ax, voxel):
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxel, 0)
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('blue')
    print(verts[faces])
    ax.add_collection3d(mesh)

import vtk
from mayavi import mlab
mlab.options.offscreen = True
def plot_mayavi(voxel):
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxel, 0)
    mlab.triangular_mesh([vert[0] for vert in verts],
                         [vert[1] for vert in verts],
                         [vert[2] for vert in verts],
                         faces)
    mlab.savefig("test.png")

#import visvis as vv
def plot_vv(voxel):
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxel, 0)
    vv.mesh(np.fliplr(verts), faces, normals, values)
    vv.use().Run()

viewpoints = [(-65,55), (-45, 75)]

paths = ["/data/bw462/3d_recon/ShapeNetVox32/02828884/10654ea604644c8eca7ed590d69b9804/model.binvox",
         "/data/bw462/3d_recon/ShapeNetVox32/02691156/110f6dbf0e6216e9f9a63e9a8c332e52/model.binvox"]
for i, path in enumerate(paths):
    voxel = read_as_3d_array(open(path, "rb")).data.squeeze()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.view_init(*viewpoints[i])
    #ax.set_axis_off()
    #ax.set_aspect('equal')
    #ax.dist = 500
    #plotMatrix(ax, voxel)
    #plot_mesh(ax, voxel)
    #plt.show()
    #plot_vv(voxel)
    plot_mayavi(voxel)
"""
for path in sys.argv[1:]:
    voxel = read_as_3d_array(open(path, "rb")).data.squeeze()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(-65, 85)
    _ = ax.voxels(voxel, color='gray')#, edgecolor='k')
    ax.set_axis_off()
    plt.savefig(path.split('/')[-2] + '.png')
"""
