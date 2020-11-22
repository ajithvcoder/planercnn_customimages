#computer vision - env 
import pyvista as pv 
from pyvista import examples

filename = '../test2/0_model.ply'

print(filename)

mesh = pv.read(filename)
cpos = mesh.plot()

plotter = pv.Plotter(off_screen= True)
plotter.add_mesh(mesh)
plotter.show(screenshot="myscreenshot.png")