TRST-MI
----------

This package contains code for minimizing the coherence of complex matrices.

The file trstmi.py contains the main script used for generating low coherence matrices. The arguments of this function take the order:

dim1 dim2 num1 num2 trials tol proc verbose

"dim1" is the lower bound on dimension, "dim2" is the upper. "num1" and "num2" serve a similar purpose except give the number of points to optimize over. "trials" gives the number of different starting random initializations to optimize over. "tol" is the stopping gradient tolerance. "proc" takes arguments cpu/gpu. "verbose" takes arguments 0/1.


This code runs on Python3. You will need PyTorch installed. If you would like to use GPU parallelization you will need a version of PyTorch installed supporting your hardware.
The trust region optimizer used in the program is not a built-in optimizer in PyTorch and comes from a public github repo. It may be installed with:

python -m pip install git+https://github.com/vchoutas/torch-trust-ncg

If you have any questions about the code, please contact: 

Josiah Park
Texas A&M University, 
Department of Mathematics, 
j.park@tamu.edu