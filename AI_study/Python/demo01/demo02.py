import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(-1,1,50)
# y = 2*x + 1

# plt.figure()
# plt.plot(x,y)
# plt.show()


x = np.linspace(-3,3,50)
y1 = 2*x + 1
y2 = x**2
plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color='red',linewidth=1.0,linestyle='--')
plt.xlim((0,2))
plt.ylim((0,3))
plt.xlabel('I am x')
plt.ylabel('i am y')
new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3], ['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])
ax = plt.gca()
ax.spines['right'].set_color('red')
ax.spines['top'].set_color('none')
plt.show()