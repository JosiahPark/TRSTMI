import torch
from torchtrustncg import TrustRegion
from torchtrustncg.utils import rosenbrock, branin
from torch import linalg as LA
import sys
import os
import numpy as np
from timeit import default_timer as timer

def copt(x, epsilon):
    dimen = int(x.size()[1]/2)
    norm = LA.vector_norm(x, dim=1, keepdim=True)
    x3 =x/norm
    x1=x3[...,:dimen] 
    x2=x3[...,dimen:]
    xxt1=torch.matmul(x1,x1.T)
    xxt2=torch.matmul(x2,x2.T)
    xxt3=torch.matmul(x2,x1.T)
    xxt4=torch.matmul(x1,x2.T)
    xxt=(xxt1+xxt2)**2+(xxt3-xxt4)**2
    xxt=torch.triu(xxt,1)
    s=torch.max(xxt)
    expxxt=torch.exp((xxt-s)/epsilon)
    u=torch.triu(expxxt,1).sum()
    f=s+epsilon*torch.log(u)
    return f
    
def coh(x):
    N = x.size()[0]
    dimen = int(x.size()[1]/2)
    norm = LA.vector_norm(x, dim=1, keepdim=True)
    x = x/norm
    x1=x[...,:dimen]
    x2=x[...,dimen:]
    xxt1=torch.matmul(x1,x1.T)
    xxt2=torch.matmul(x2,x2.T)
    xxt3=torch.matmul(x2,x1.T)
    xxt4=torch.matmul(x1,x2.T)
    xxt=(xxt1+xxt2)**2+(xxt3-xxt4)**2
    return torch.max(torch.triu(xxt,1))**(1/2)


def min_coh(N, d, epsilon, x0):
    
    x0.grad = None
    optimizer = TrustRegion([x0], opt_method='cg')
    loss=copt(x0,epsilon)
    loss1=0
    
    def closure(backward=True):
        if backward:
            optimizer.zero_grad(set_to_none=True)
        loss = copt(x0,epsilon)
        if backward:
            loss.backward(create_graph=True)
        return loss
    
    start = timer()
    
    for l in range(20*N):
        loss = optimizer.step(closure)
        if torch.abs(loss-loss1)<1e-10 and (l + 1) % 5 == 0:
            break
        if torch.norm(x0.grad).item() < gtol:
            break
        if torch.norm(optimizer.param_step, dim=-1).lt(gtol).all():
            break
        loss1 = loss
        if (l + 1) % 20 == 0:
            print(f'TR iter:{l+1}, loss={loss1.item()}')
        
    end = timer()
    elapsed = end-start
    print(f'Stage {-int(np.floor(np.log(epsilon)/np.log(10)))} Complete: (d={d}, N={N})')
    print(f'Coherence: {coh(x0)} ')
    return x0, elapsed

def save_run(d, N, coh, x, eps, trial, elapsed):
    os.makedirs('optc-64e6t-d-{0}'.format(d),exist_ok=True)
    os.makedirs('optc-64e6t-d-{0}/optc-d-{0}-n-{1}'.format(d,N),exist_ok=True)
    np.savetxt('optc-64e6t-d-{0}/optc-d-{0}-n-{1}/optc-d-{0}-n-{1}-coh-{2}-eps-{3}-trial-{4}-time-{5}.txt'.format(d,N,coh,eps,trial,elapsed),x.cpu().detach().numpy(),delimiter=' ', newline='\n')
    
def run_trials(d1,d2,n1,n2, eps_pows, n_trials):
    for d in range(d1,d2):
        for N in range(n1,n2):       
            for trial in range(1, n_trials+1):
                print('')
                print('')
                x0 = torch.randn((N,2*d),requires_grad=True,device=device,dtype=torch.float64)
                for epsilon, eps_pow in zip(10.**(-eps_pows), eps_pows):
                    x0, elapsed = min_coh(N, d, epsilon, x0)
                    save_run(d, N, coh(x0), x0, eps_pow, trial, elapsed)
                
        

if len(sys.argv) <5:
 print('Error: trstmi takes 4 arguments or 8. The full input parameters (in order) are dim1 dim2 num1 num2 trials tol proc. "dim1" is the lower bound on dimension, "dim2" is the upper. "num1" and "num2" serve a similar purpose except give the number of points to optimize over. "trials" gives the number of different starting random initializations to optimize over. "tol" is the stopping gradient tolerance. "proc" takes arguments cpu/gpu. "verbose" takes arguments 0/1.')
 exit(1)
 
dim1=int(float(eval(sys.argv[1])))
dim2=int(float(eval(sys.argv[2])))
num1=int(float(eval(sys.argv[3])))
num2=int(float(eval(sys.argv[4])))


if len(sys.argv) >5 and len(sys.argv) <9:
 print('Error: trstmi takes 4 arguments or 8. The full input parameters (in order) are dim1 dim2 num1 num2 trials tol proc. "dim1" is the lower bound on dimension, "dim2" is the upper. "num1" and "num2" serve a similar purpose except give the number of points to optimize over. "trials" gives the number of different starting random initializations to optimize over. "tol" is the stopping gradient tolerance. "proc" takes arguments cpu/gpu. "verbose" takes arguments 0/1.')
 exit(1)


if len(sys.argv) > 5:
 trials=int(float(eval(sys.argv[5])))
 tol=float(eval(sys.argv[6]))
 proc=sys.argv[7]
 verbose=sys.argv[8]
else:
 trials=100
 tol=1e-6
 proc=cpu
 verbose=0


torch.set_printoptions(precision=24)
device = torch.device(proc)
torch.set_default_dtype(torch.float64)
gtol=tol

run_trials(dim1,dim2,num1,num2, torch.arange(1,16), trials)