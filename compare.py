import autodiff
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) + np.cos(x) + x

h = 1e-9
xs = np.linspace(-100, 100, 1000)
errornd = []
errorad = []

for x in xs:

    nd = (f(x+h) - f(x)) / h

    a  =  autodiff.vec([x])
    Q = a.sin() + a.cos() + a
    Q.backward()
    ad = a.grad()[0]
    gold = (a.cos()-a.sin()+1).values()[0]


    errornd.append(np.abs(nd-gold))
    errorad.append(np.abs(ad-gold))

plt.figure()
plt.plot(xs, errornd, label='numerical diff')
plt.plot(xs, errorad, label='automatic diff')
plt.xlabel('x')
plt.ylabel('error')
plt.legend()
plt.savefig('comp.png')
plt.close()