#Python libraries
import math
import numpy as np
import matplotlib.pyplot as plt

#Parameters
A = 1
upper = 0.9
alf = 0.3
n = 0.04
rho = 0.05
delta = 0.1
q = 0.1
k_st = 0.1

#Step 1: Numerical estimation of the steady state

#z * (alf * A * k**(alf - 1) + rho - alf * A * alf * k**(alf - 1)) - q * alf - 1,
#alf * A * k**alf - delta * k - k/z

k_0 = ((q*alf*delta + rho + delta)/(A*alf**2*(q + 1)))**(1/(alf-1))
z_0 = (alf*(q + 1))/(rho + delta*(1 - alf))

print(k_0, z_0)

#Step 2: Calculating the elements and determinant of the matrix A

#dz/dt = z * (alf * A * k**(alf - 1) + rho - alf * A * alf * k**(alf - 1)) - q * alf - 1,
#dk/dt = alf * A * k**alf - delta * k - k/z

a11 = alf**2*A*k_0**(alf - 1) - delta - 1/z_0
a12 = k_0/z_0**2
a21 = -z_0 * A * alf * (alf - 1)**2 * k_0**(alf - 2)
a22 = alf * A * (1 - alf) * k_0**(alf - 1) + rho

matrix_A = np.array([[a11, a12], [a21, a22]])
d = np.linalg.det(matrix_A)
print(matrix_A)

#Step 4: k_e, z_e
eps = 0.00001
eigenvalues, eigenvectors = np.linalg.eig(np.array([[a11, a12], [a21, a22]]))
k_eps = k_0 + eigenvectors[0][0] * eps
z_eps = z_0 + eigenvectors[0][1] * eps
print(eigenvalues)
print(eigenvectors)
print(k_eps, z_eps)

#Step 5: Runge–Kutta method

def dzdk(k, z): #dz/dk
    return (z * (alf * A * k**(alf - 1) + rho - alf * A * alf * k**(alf - 1)) - q * alf - 1) / (alf * A * k**alf - delta * k - k/z)

def RK4(k, z, func):
    # Подсчет инкрементов
    K1 = func(k, z)
    K2 = func(k + dk/2, z + K1 * dk/2)
    K3 = func(k + dk/2, z + K2 * dk/2)
    K4 = func(k + dk, z + K3 * dk)

    # Обновляем значение
    return dk/6 * (K1 + 2*K2 + 2*K3 + K4)

# Задаем начальные условия
z = z_eps

# Задаем временной интервал и шаг
k = k_eps
dk = -0.000001

# Создаем списки для сохранения значений
k_val = []
z_val = []
l1_val = []
l2_val = []

while k >= k_st:

    z += RK4(k, z, dzdk)
    #print(k, z)

    # Сохраняем текущие значения
    k_val.append(k)
    z_val.append(z)
    l1_val.append(k/(A*(k)**alf))
    l2_val.append(k/(A*(k)**alf * (1 - upper)))

    # Обновляем время
    k += dk

# Рисуем и сохраняем график
plt.plot(list(reversed(k_val)), list(reversed(z_val)), label='z(k)', color='r')
plt.plot(k_eps, z_eps, marker='*', color='r')
plt.text(x=k_eps, y=z_eps - 0.15, s="(k_eps, z_eps)",
    fontdict=dict(fontsize=10, fontweight="bold"), ## Font settings
    backgroundcolor="white", alpha=0.5,
    ha="center", va="bottom", )
plt.plot(k_st, z_val[-1], marker='*', color='r')
plt.text(x=k_st + 0.06, y=z_val[-1], s="(k(0), z(0))",
    fontdict=dict(fontsize=10, fontweight="bold"), ## Font settings
    backgroundcolor="white", alpha=0.5,
    ha="center", va="bottom", )


plt.axvline(x=k_st, color='y', label='k(0)', linestyle='dashed')
plt.xlabel('k')
plt.ylabel('z')
plt.legend()
plt.savefig('dzdk.png')
plt.show()

def differ(k, z): #dk/dt, dz/dt
    return alf * A * k**alf - delta * k - k/z, z * (alf * A * k**(alf - 1) + rho - alf * A * alf * k**(alf - 1)) - q * alf - 1

def RK4_vec2(k, z):
    # Подсчет инкрементов
    K1 = differ(k, z)
    K2 = differ(k + K1[0] * dt/2, z + K1[1] * dt/2)
    K3 = differ(k + K2[0] * dt/2, z + K2[1] * dt/2)
    K4 = differ(k + K3[0] * dt, z + K3[1] * dt)

    # Обновляем значение
    return dt/6 * (K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), dt/6 * (K1[1] + 2*K2[1] + 2*K3[1] + K4[1])

# Задаем начальные условия
zt = z_val[-1]
kt = k_st
print(z_val[-1])

# Задаем временной интервал и шаг
t = 0
t_end = 30
dt = 0.0001

# Создаем списки для сохранения значений
time = []
kt_val = []
zt_val = []
u_val = []

k_max = 0
z_max = 0
t_max_k = 0
t_max_z = 0


while t < t_end:

    inc = RK4_vec2(kt, zt)
    kt += inc[0]
    zt += inc[1]

    if kt > k_max:
      k_max = kt
      t_max_k = t
    if zt > z_max:
      z_max = zt
      t_max_z = t


    # Сохраняем текущие значения
    time.append(t)
    kt_val.append(kt)
    zt_val.append(zt)
    u_val.append(1 - kt / (zt * A * kt**alf))

    # Обновляем время
    t += dt


print(k_max, t_max_k)
print(z_max, t_max_z)
# Рисуем и сохраняем график
# plt.plot(time, K_values, label='K(t)')
# plt.plot(time, Y_values, label='Y(t)')
#plt.plot(time, zt_val, label='z(t)', color='r')
plt.plot(time, kt_val, label='k(t)', color='g')
plt.axhline(y=k_0, color='r', label='k*', linestyle='dashed')
plt.xlabel('t')
plt.ylim(-0.1,0.7)
#plt.title('v = 0.2')
plt.legend()
plt.savefig('k(t).png')
plt.show()

plt.plot(time, zt_val, label='z(t)', color='b')
plt.axhline(y=z_0, color='r', label='z*', linestyle='dashed')
plt.xlabel('t')
plt.legend()
plt.ylim(0.5,3)
plt.savefig('z(t).png')
plt.show()

plt.plot(time, u_val, label='u(t)', color='y')
plt.xlabel('t')
#plt.title('v = 0.2')
plt.legend()
#plt.ylim(-0.1,1)
plt.savefig('u(t).png')
plt.show()
