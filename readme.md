Lid Driven Cavity Problem
=========================

Solver for the Lid Driven Cavity Problem using several frameworks and libraries.
Velocities U and V are solved using the Navier Stokes equations in an staggered grid.
The Conservation of Mass is also solved as an equation for pressure.
Everything is solved together (coupled) using a nonlinear solver.

Example results (100x100 mesh, Re=1000.0):

![streamlines](./experiments/100x100_re1000_streamlines.png)
![quiver](./experiments/100x100_re1000_quiver.png)
