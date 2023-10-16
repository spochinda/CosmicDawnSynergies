import matplotlib.pyplot as plt
import numpy as np
loss = np.loadtxt("data/globalemu/emulator4/results/loss_history.txt")
x=range(loss.size)

fig,ax = plt.subplots(nrows=1,ncols=1)
ax.plot(x,loss,marker="o",ls="solid")
plt.savefig("test.png")