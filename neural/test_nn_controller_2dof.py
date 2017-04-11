"""
Simulator for a two-linkage robotic manipulator
used for testing the 2dof neural net controller.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nn_controller_2dof import NN_controller

################################################# PHYSICAL PARAMETERS

# Simulation duration, timestep, and animation
T = 40  # s
dt = 0.001  # s
framerate = 60  # fps
outline_path = False  # should path outline be shown on animation

# Initial condition
q = np.deg2rad([-90, 0, 0, 0])  # [rad, rad, rad/s, rad/s]

# Link lengths
L = [1, 0.5]  # m

# Link masses
m = [5, 3]  # kg

# Local gravity
g = 9.81  # m/s^2

# Joint damping
d = [0.05, 0.05]  # (N*m)/(rad/s)

# Joint friction
b = [1, 1]  # N*m
c = [2, 2]  # s/rad

# Actuator limits
umax = [250, 30]  # N*m

# Vibration noise
vibe_mean = [0, 0]  # N*m
vibe_stdv = [0, 0]  # N*m

# Sensor noise
sensor_mean = [0, 0, 0, 0]  # [rad, rad, rad/s, rad/s]
sensor_stdv = [0, 0, 0, 0]  # [rad, rad, rad/s, rad/s]

################################################# CONTROL SYSTEM PARAMETERS

# Proportional gains
kp = [100, 100]  # (N*m)/(rad)

# Derivative gains
kd = [100, 100]  # (N*m)/(rad/s)

# Learning gains
n = 10  # number of neurons
kv = 10 * np.ones((len(q)+1, len(q)+1))
kw = 10 * np.ones((n+1, n+1))

# Path to track
path_type = 'train'  # 'waypoint', 'random', 'train', or 'cycle'
target = np.deg2rad([70, 45]) # m, or rad (for 'train' and 'cycle')
vmax = [np.pi, np.pi]  # rad/s
amax = [5, 1] # rad/s^2

# Initialize controller
controller = NN_controller(dt, q, target, path_type, kp, kd, n, kv, kw, umax, vmax, amax)

################################################# EQUATIONS OF MOTION

def dynamics(q, u):
	"""
	Returns state derivative (qdot).
	Takes control input (u) and current state (q).

	"""
	# Externally set parameters
	global L, m, g, b, c, umax

	# Mass matrix M(q)
	M = np.zeros((2, 2))
	M[0, 0] = (m[0]+m[1])*L[0]**2 + m[1]*L[1]**2 + 2*m[1]*L[0]*L[1]*np.cos(q[1])
	M[0, 1] = m[1]*L[1]**2 + m[1]*L[0]*L[1]*np.cos(q[1])
	M[1, 0] = M[0, 1]  # symmetry
	M[1, 1] = m[1]*L[1]**2

	# Centripetal and coriolis vector V(q)
	V = np.array([
	              -m[1]*L[0]*L[1]*(2*q[2]*q[3]+q[3]**2)*np.sin(q[1]),
	               m[1]*L[0]*L[1]*q[2]**2*np.sin(q[1])
				])

	# Gravity vector G(q)
	G = np.array([
	              g*(m[0]+m[1])*L[0]*np.cos(q[0]) + m[1]*g*L[1]*np.cos(q[0]+q[1]),
	              m[1]*g*L[1]*np.cos(q[0]+q[1])
				])

	# Joint damping D(q)
	D = np.array([
	              d[0]*q[2],
	              d[1]*q[3]
				])

	# Joint friction
	F = np.array([
	              b[0]*np.tanh(c[0]*q[2]),
	              b[1]*np.tanh(c[1]*q[3])
				])

	# Vibration noise introduced in an industrial environment
	f = vibe_mean + vibe_stdv*np.random.randn(2)

	# Store internal dynamics for viewing
	global M_store, V_store, G_store, F_store
	M_store = M
	V_store = V
	G_store = G
	F_store = F

	# Actuator saturation
	for i, mag in enumerate(abs(u)):
		if mag > umax[i]:
			u[i] = umax[i] * np.sign(u[i])

	# [theta1dot, theta2dot] = [w1, w2]   and   [w1dot, w2dot] = (M^-1)*(u-V-G-D-F)
	return np.concatenate((q[2:], np.linalg.inv(M).dot(u + f - V - G - D - F)))


def kinem_forward(q, L):
	"""
	Returns the state of the end effector (x = [px, py, vx, vy]).
	Takes the current joint state (q) and link lengths (L).

	"""
	return np.array([
				     L[0]*np.cos(q[0]) + L[1]*np.cos(q[0]+q[1]),
				     L[0]*np.sin(q[0]) + L[1]*np.sin(q[0]+q[1]),
				     -L[0]*np.sin(q[0])*q[2] - L[1]*np.sin(q[0]+q[1])*(q[2]+q[3]),
				     L[0]*np.cos(q[0])*q[2] + L[1]*np.cos(q[0]+q[1])*(q[2]+q[3])
				   ])

################################################# SIMULATION

# Define time domain
t_arr = np.arange(0, T, dt)

# Preallocate results memory
q_history = np.zeros((len(t_arr), 4))
x_history = np.zeros((len(t_arr), 4))
qref_history = np.zeros((len(t_arr), 4))
xref_history = np.zeros((len(t_arr), 4))
target_history = np.zeros((len(t_arr), 2))
u_history = np.zeros((len(t_arr), 2))
y_history = np.zeros((len(t_arr), 2))
dyn_history = np.zeros((len(t_arr), 2))

# Keep some dynamics internals too
M_store = []
V_store = []
G_store = []
F_store = []

# Integrate dynamics using zero-order forward stepping
for i, t in enumerate(t_arr):

	# Controller's decision
	sensor_noise = sensor_mean + sensor_stdv*np.random.randn(4)
	u = controller.get_effort(q + sensor_noise, dt)

	# Dynamics at this instant
	qdot = dynamics(q, u)
	
	# Record this instant
	q_history[i, :] = q
	x_history[i, :] = kinem_forward(q, L)
	qref_history[i, :] = controller.qref
	xref_history[i, :] = kinem_forward(controller.qref, L)
	target_history[i, :] = controller.target
	u_history[i, :] = u
	y_history[i, :] = controller.y
	dyn_history[i, :] = M_store.dot(controller.aref) + M_store.dot(controller.kr*(controller.qref[2:] - q[2:])) + V_store + G_store + F_store

	# Quit early if something breaks
	if controller.kill:
		break

	# Modify any time-varying parameters
	pass

	# Step forward, qnext = qlast + qdot*dt
	q = q + qdot*dt

################################################# VISUALIZATION

# Figure for joint space results
fig1 = plt.figure()
fig1.suptitle('Joint Space', fontsize=20)

# Plot joint angle 1
ax1 = fig1.add_subplot(2, 3, 1)
ax1.set_ylabel('Angle 1 (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 0]), 'k',
		 t_arr, np.rad2deg(qref_history[:, 0]), 'g--')
ax1.grid(True)

# Plot joint angle 2
ax1 = fig1.add_subplot(2, 3, 2)
ax1.set_ylabel('Angle 2 (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 1]), 'k',
		 t_arr, np.rad2deg(qref_history[:, 1]), 'g--')
ax1.grid(True)

# Plot control efforts
ax1 = fig1.add_subplot(2, 3, 3)
ax1.set_ylabel('Torque (N*m)', fontsize=16)
ax1.plot(t_arr, u_history[:, 0], 'b',
		 t_arr, u_history[:, 1], 'g')
ax1.grid(True)

# Plot joint velocity 1
ax1 = fig1.add_subplot(2, 3, 4)
ax1.set_ylabel('Velocity 1 (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 2]), 'k',
		 t_arr, np.rad2deg(qref_history[:, 2]), 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot joint velocity 1
ax1 = fig1.add_subplot(2, 3, 5)
ax1.set_ylabel('Velocity 2 (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 3]), 'k',
		 t_arr, np.rad2deg(qref_history[:, 3]), 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot adaptive estimates
ax1 = fig1.add_subplot(2, 3, 6)
ax1.set_ylabel('NN Estimation Error', fontsize=16)
ax1.plot(t_arr, dyn_history[:, 0]-y_history[:, 0], 'b',
		 t_arr, dyn_history[:, 1]-y_history[:, 1], 'g')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

plt.show()

# Plot for repetative learning results if applicable
fig1a = plt.figure()
fig1a.suptitle('NN Learning Evolution', fontsize=20)
ax1a = fig1a.add_subplot(1, 2, 1)
ax1a.set_xlabel('Time (s)')
ax1a.set_ylabel('Link 1 Dynamics Estimate (N*m)')
ax1a.plot(t_arr, y_history[:, 0], 'k', t_arr, dyn_history[:, 0], 'g--')
ax1a = fig1a.add_subplot(1, 2, 2)
ax1a.set_xlabel('Time (s)')
ax1a.set_ylabel('Link 2 Dynamics Estimate (N*m)')
ax1a.plot(t_arr, y_history[:, 1], 'k', t_arr, dyn_history[:, 1], 'g--')

plt.show()

# Create figure for end effector results
fig2 = plt.figure()
fig2.suptitle('End Effector Space', fontsize=20)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_xlabel('x-position (m)')
ax2.set_ylabel('y-position (m)')
ax2.set_xlim([-np.sum(L), np.sum(L)])
ax2.set_ylim([-np.sum(L), np.sum(L)])

# Plot parametric end effector position
ax2.plot(x_history[:, 0], x_history[:, 1], 'k', xref_history[:, 0], xref_history[:, 1], 'g--')
ax2.scatter(x_history[0, 0], x_history[0, 1], color='r', s=50)
if path_type not in ['train', 'cycle']:
	ax2.scatter(target_history[:, 0], target_history[:, 1], color='g', s=50)
ax2.grid(True)

plt.show()

# Figure for animation
fig3 = plt.figure()
fig3.suptitle('Evolution')
ax3 = fig3.add_subplot(1, 1, 1)
ax3.set_xlabel('- World X +')
ax3.set_ylabel('- World Y +')
ax3.set_xlim([-np.sum(L)-1, np.sum(L)+1])
ax3.set_ylim([-np.sum(L)-1, np.sum(L)+1])
ax3.grid(True)

# Position of intermediate joint during motion
elb_history = np.concatenate(([L[0]*np.cos(q_history[:, 0])], [L[0]*np.sin(q_history[:, 0])])).T

# Lines for representing the links and points for joints
lthick = 3
pthick = 25
link1 = ax3.plot([0, elb_history[0, 0]], [0, elb_history[0, 1]], color='k', linewidth=lthick)
link2 = ax3.plot([elb_history[0, 0], x_history[0, 0]], [elb_history[0, 1], x_history[0, 1]], color='k', linewidth=lthick)
end = ax3.scatter(x_history[0, 0], x_history[0, 1], color='k', s=pthick*m[1], zorder=2)
elb = ax3.scatter(elb_history[0, 0], elb_history[0, 1], color='k', s=pthick*m[0], zorder=2)
bse = ax3.scatter(0, 0, color='k', s=pthick)

# Desired trajectory curve and tracking point
pointref = ax3.scatter(xref_history[0, 0], xref_history[0, 1], color='g', s=pthick*m[1], zorder=3)
if path_type not in ['train', 'cycle']:
	pointtar = ax3.scatter(target[0], target[1], color='r', s=50, zorder=4)

# Plot entirety of actual trajectory
if outline_path:
	outline = ax3.plot(x_history[:, 0], x_history[:, 1], 'k--', linewidth=1)
	outlineref = ax3.plot(xref_history[:, 0], xref_history[:, 1], 'g--', linewidth=lthick/3)

# Function for updating the animation frame
def update(arg, ii=[0]):

	i = ii[0]  # don't ask...

	if np.isclose(t_arr[i], np.around(t_arr[i], 1)):
		fig3.suptitle('Evolution (Time: {})'.format(t_arr[i]), fontsize=24)

	link1[0].set_data([0, elb_history[i, 0]], [0, elb_history[i, 1]])
	link2[0].set_data([elb_history[i, 0], x_history[i, 0]], [elb_history[i, 1], x_history[i, 1]])
	end.set_offsets((x_history[i, 0], x_history[i, 1]))
	elb.set_offsets((elb_history[i, 0], elb_history[i, 1]))

	pointref.set_offsets((xref_history[i, 0], xref_history[i, 1]))
	if path_type not in ['train', 'cycle']:
		pointtar.set_offsets((target_history[i, 0], target_history[i, 1]))

	ii[0] += int(1 / (dt * framerate))
	if ii[0] >= len(t_arr):
		print("Resetting animation!")
		ii[0] = 0

	return [link1, link2, end, elb, pointref]

# Run animation
ani = animation.FuncAnimation(fig3, func=update, interval=dt*1000)
print("\nRemember to keep the diplay window aspect ratio square.\n")
plt.show()
