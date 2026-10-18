import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

#Parameters
v0 = 1
v_temp = 0.05
Mode = 1
dt = 0.1
N_Steps = 400
grid_no = Mode * 64
N = Mode * 20000

a = 0.612
k = a / v0
L = 2 * np.pi * Mode / k
dx = L / grid_no
Q = -L / N
m = L / N

def particles():
    beam_a = np.linspace(0, L, N // 2, endpoint = False)
    beam_b = np.linspace(0, L, N//2, endpoint = False)
    x = np.concatenate((beam_a, beam_b))
    
    vel_a = np.full(N//2, v0)
    vel_b = np.full(N//2, -v0)
    v = np.concatenate((vel_a, vel_b))
    
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(N)
    v = v + v_temp * noise
    
    x = (x + 0.001 * np.cos(2 * np.pi * x / L)) % L
    
    return x, v

x, v = particles()

def background_charge(x):
    s = x / dx
    i = np.floor(s).astype(int)
    frac = s - i
    
    w_left = 1 - frac
    w_right = frac
    
    left = i % grid_no
    right = (i + 1) % grid_no
    
    l = np.bincount(left, weights = w_left, minlength = grid_no)
    r = np.bincount(right, weights = w_right, minlength = grid_no)
    count = l + r
    
    rho_electrons = Q * count / dx
    rho_ions = -Q * N / L
    rho = rho_electrons + rho_ions
    
    return rho

rho = background_charge(x)

def solve_field(rho):
    k = 2 * np.pi * np.fft.fftfreq(grid_no, d = dx)
    rho_k = np.fft.fft(rho)
    
    E_k = np.zeros_like(rho_k)
    E_k[1:] = rho_k[1:] / (k[1:] * 1j)
    E = np.real(np.fft.ifft(E_k))
    
    return E


def interpolate_field(E, x):
    s = x / dx
    i = np.floor(s).astype(int)
    frac = s - i
    
    left = i % grid_no
    right = (i + 1) % grid_no
    
    return E[left] * (1 - frac) + E[right] * frac

def step(x, v):
    rho = background_charge(x)
    E = solve_field(rho)
    Ep = interpolate_field(E, x)
    
    v = v + (Q / m) * Ep * dt
    x = (x + v * dt) % L 
    
    return x, v, E


#loop
field_energy = np.zeros(N_Steps)
snap_x, snap_v, snap_t = [], [], []


for n in range(N_Steps):
    x, v, E = step(x,v)
    field_energy[n] = 0.5 * np.sum(E**2) * dx
    
    if n % 2 == 0:
        snap_x.append(x.copy())
        snap_v.append(v.copy())
        snap_t.append(n * dt)
        
t = np.arange(N_Steps) * dt


#plot
plt.semilogy(t, field_energy)
plt.xlabel("t")
plt.ylabel("Field Energy")
plt.show()


snapshots = [0, len(snap_t) // 4, len(snap_t) // 2, len(snap_t) - 1]
fig, axes = plt.subplots(1, 4, figsize = (16, 4))

for ax, j in zip(axes, snapshots):
    ax.scatter(snap_x[j], snap_v[j], s = 0.2)
    ax.set_title(f"t = {snap_t[j]:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("v")

plt.show()

#Animation
fig, ax = plt.subplots()
sc = ax.scatter(snap_x[0], snap_v[0], s = 0.5, c = snap_v[0], cmap = "coolwarm", vmin = -2, vmax = 2)
ax.set_xlim(0, L)
ax.set_ylim(-2.5, 2.5)
title = ax.set_title("")

def update(frame):
    sc.set_offsets(np.c_[snap_x[frame], snap_v[frame]])
    sc.set_array(snap_v[frame])
    title.set_text(f"t = {snap_t[frame]:.1f}")
    return sc, title

anim = FuncAnimation(fig, update, frames = len(snap_t), interval = 50)
anim.save(r"C:\Users\bobrh\Desktop\two_stream.gif", writer="pillow")

plt.show()




    
    







    
    
    
    