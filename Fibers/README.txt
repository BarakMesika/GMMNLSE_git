BuildFiber

This folder contains the tools required to build a fiber from the parameters of an index profile, for any of the three simulation methods. For all methods, solve_for_modes is needed to solve for the modes of the fiber. For the 3D code and the RC code this is all that is needed to perform simulations. For the MM code, the dipsersion and mode overlap tensors also need to be calculated with "calc_dispersion.m" and "calc_SRSK_tensors."

It is suggested that the fiber modes and corresponding dispersion coefficients and overlap tensors be built locally in this folder and then copied to whichever simulation folder you are using.

To go through one example of building a fiber from scratch, run the RUNME.m script. This builds a fiber that supports 6 modes and demonstrates how each of the files work to produce all the information needed for simulations.