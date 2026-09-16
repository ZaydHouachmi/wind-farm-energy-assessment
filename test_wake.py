import numpy as np
from wake_model import SimpleWakeModel, JensenWakeModel

sm = SimpleWakeModel(alpha=0.05, beta=0.5)
jm = JensenWakeModel(wake_decay=0.05)

xi = np.array([[1.0, 2.0, -1.0], [0.5, 5.0, 10.0]])
eta = np.zeros_like(xi)

print('xi shape:', xi.shape)
print('SimpleWakeModel deficit:')
print(sm.wake_deficit(xi, eta, 90.0))
print('\nJensenWakeModel deficit:')
print(jm.wake_deficit(xi, eta, 90.0))
