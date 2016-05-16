"""
Simulator for a two-linkage robotic manipulator.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from control_2link import Controller

################################################# PHYSICAL PARAMETERS

# Simulation duration, timestep, and animation
T = 20  # s
dt = 0.001  # s
framerate = 60  # fps
animate_adapt = True  # should adaptation be shown on animation
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
b = [0.1, 0.1]  # N*m
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

# Adaptive gains, stack size, and initial estimate
kg = 'LS'
ku = 5*np.ones(5)
kf = 0
window = np.inf  # s
history_size = 50
adapt0 = [0, 0, 0, 0, 0]  # [(m1+m2)*g*L1, (m1+m2)*L1^2, m2*g*L2, m2*L1*L2, m2*L2^2]_est

# Path to track
path_type = 'random'  # 'waypoint', 'random', 'train', or 'cycle'
target = np.deg2rad([70, 45]) # m, or rad (for 'train' and 'cycle')
vmax = [np.pi, np.pi]  # rad/s
amax = [5, 1] # rad/s^2

# Initialize controller
controller = Controller(dt, q, target, path_type, kp, kd, kg, ku, kf, umax, vmax, amax, history_size, window, adapt0)

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

	# Record joint friction for examining the repetitive disturbance
	global unstruct_history, i
	unstruct_history[i, :] = F + D

	# Vibration noise introduced in an industrial environment
	f = vibe_mean + vibe_stdv*np.random.randn(2)

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
u_history = np.zeros((len(t_arr), 2))
adapt_history = np.zeros((len(t_arr), 5))
Lest_history = np.zeros((len(t_arr), 2))
mest_history = np.zeros((len(t_arr), 2))
xest_history = np.zeros((len(t_arr), 4))
target_history = np.zeros((len(t_arr), 2))
unstruct_history = np.zeros((len(t_arr), 2))  # recorded in dynamics function
rep_history = np.zeros((len(t_arr), 2))

# Integrate dynamics using first-order forward stepping
for i, t in enumerate(t_arr):

	# Controller's decision
	sensor_noise = sensor_mean + sensor_stdv*np.random.randn(4)
	u = controller.get_effort(q + sensor_noise, dt)
	
	# Record this instant
	q_history[i, :] = q
	x_history[i, :] = kinem_forward(q, L)
	qref_history[i, :] = controller.qref
	xref_history[i, :] = kinem_forward(controller.qref, L)
	u_history[i, :] = u
	adapt_history[i, :] = controller.adapt
	target_history[i, :] = controller.target
	rep_history[i, :] = controller.rep

	# Quit early if something breaks
	if controller.kill:
		break

	# Record approximately what the controller thinks the chain is
	Lest_history[i, :] = controller.Lest
	mest_history[i, :] = controller.mest
	xest_history[i, :] = kinem_forward(q, Lest_history[i, :])

	# Modify any time-varying parameters
	pass

	# Step forward, qnext = qlast + qdot*dt
	q = q + dynamics(q, u)*dt

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
adapt_true = [np.sum(m)*g*L[0], np.sum(m)*L[0]**2, m[1]*g*L[1], m[1]*np.product(L), m[1]*L[1]**2]
ax1 = fig1.add_subplot(2, 3, 6)
ax1.set_ylabel('Adaptive Estimates', fontsize=16)
ax1.plot(t_arr, adapt_history[:, 0], 'b',
		 t_arr, adapt_history[:, 1], 'g',
		 t_arr, adapt_history[:, 2], 'r',
		 t_arr, adapt_history[:, 3], 'c',
		 t_arr, adapt_history[:, 4], 'm',
		 [0, t_arr[-1]], adapt_true[0]*np.ones(2), 'b--',
		 [0, t_arr[-1]], adapt_true[1]*np.ones(2), 'g--',
		 [0, t_arr[-1]], adapt_true[2]*np.ones(2), 'r--',
		 [0, t_arr[-1]], adapt_true[3]*np.ones(2), 'c--',
		 [0, t_arr[-1]], adapt_true[4]*np.ones(2), 'm--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

print("\nInitial adaptation: {}".format(np.round(adapt0, 1)))
print("Final adaptation:     {}".format(np.round(controller.adapt, 1)))
print("True values:          {}\n".format(np.round(adapt_true, 1)))
plt.show()

# Plot for repetative learning results if applicable
if controller.use_RL:
	fig1a = plt.figure()
	fig1a.suptitle('Repetative Learning Evolution', fontsize=20)
	ax1a = fig1a.add_subplot(1, 2, 1)
	ax1a.set_xlabel('Time (s)')
	ax1a.set_ylabel('Link 1 Disturbance Estimate (N*m)')
	ax1a.plot(t_arr, rep_history[:, 0], 'k', t_arr, unstruct_history[:, 0], 'g--')
	ax1a = fig1a.add_subplot(1, 2, 2)
	ax1a.set_xlabel('Time (s)')
	ax1a.set_ylabel('Link 2 Disturbance Estimate (N*m)')
	ax1a.plot(t_arr, rep_history[:, 1], 'k', t_arr, unstruct_history[:, 1], 'g--')
	
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

# Represent approximately what the controller thinks the chain looks like
if animate_adapt:
	elbest_history = np.concatenate(([Lest_history[:, 0]*np.cos(q_history[:, 0])], [Lest_history[:, 0]*np.sin(q_history[:, 0])])).T
	link1est = ax3.plot([0, elbest_history[0, 0]], [0, elbest_history[0, 1]], color='b', linewidth=lthick/3)
	link2est = ax3.plot([elbest_history[0, 0], xest_history[0, 0]], [elbest_history[0, 1], xest_history[0, 1]], color='b', linewidth=lthick/3)
	endest = ax3.scatter(xest_history[0, 0], xest_history[0, 1], color='b', s=pthick*mest_history[0, 1], zorder=1)
	elbest = ax3.scatter(elbest_history[0, 0], elbest_history[0, 1], color='b', s=pthick*mest_history[0, 0], zorder=1)

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

	if animate_adapt:
		link1est[0].set_data([0, elbest_history[i, 0]], [0, elbest_history[i, 1]])
		link2est[0].set_data([elbest_history[i, 0], xest_history[i, 0]], [elbest_history[i, 1], xest_history[i, 1]])
		endest.set_offsets((xest_history[i, 0], xest_history[i, 1]))
		endest.set_sizes([pthick*mest_history[i, 1]])
		elbest.set_offsets((elbest_history[i, 0], elbest_history[i, 1]))
		elbest.set_sizes([pthick*mest_history[i, 0]])

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
