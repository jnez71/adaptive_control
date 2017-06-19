"""
Simulation to test a controller
on Fossen's dynamic model of a boat.

See:
T. I. Fossen, Handbook of Marine Craft Hydrodynamics
and Motion Control. Wiley, 2011. Chapter 13.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from control_boat import Controller

################################################# PHYSICAL PARAMETERS

# Simulation duration, timestep, and animation
T = 40  # s
dt = 0.002  # s
framerate = 20  # fps
outline_path = True  # show path outline on animation?
speedup = 1  # kinda makes the playback a little faster
store_data = False  # should data be stored into a .mat?

# Initial condition
q = np.array([3.1, 0, 1.6, 4.4, 0, 0.95])  # [m, m, rad, m/s, m/s, rad/s]

# Boat inertia and center of gravity
m = 1000  # kg
Iz = 1500  # kg*m^2
xg = 0.1  # m

# Fluid inertial effects
wm_xu = -0.025*m  # kg
wm_yv = -0.25*m  # kg
wm_yr = -0.25*m*xg  # kg*m
wm_nr = -0.25*Iz  # kg*m^2

# Drag
d_xuu = 0.25 * wm_xu  # N/(m/s)^2
d_yvv = 0.25 * wm_yv # N/(m/s)^2
d_nrr = 0.25 * (wm_nr + wm_yr) # (N*m)/(rad/s)^2

# Cross-flow
d_yrr = 0.25 * wm_yr # N/(rad/s)^2
d_yrv = 0.25 * wm_yr  # N/(m*rad/s^2)
d_yvr = 0.25 * wm_yv  # N/(m*rad/s^2)
d_nvv = 0.25 * d_yvv # (N*m)/(m/s)^2
d_nrv = 0.25 * d_yrv # (N*m)/(m*rad/s^2)
d_nvr = 0.25 * (wm_nr + wm_yv) # (N*m)/(m*rad/s^2)

# Thrust limits
umax = [np.inf, np.inf, np.inf]  # [N, N, N*m]

# Sensor noise
noise_mean = [0, 0, 0, 0, 0, 0]  # [m, m, rad, m/s, m/s, rad/s]
noise_stdv = [0, 0, 0, 0, 0, 0] # [m, m, rad, m/s, m/s, rad/s]
# [0.05, 0.05, np.deg2rad(0.8), 0.05, 0.05, np.deg2rad(2)]

################################################# CONTROL SYSTEM PARAMETERS

# Proportional and derivative gains
kp = np.array([1000, 1000, 3000])  # [N/m, N/m, (N*m)/rad]
kd = np.array([1000, 1000, 3000])  # [N/(m/s), N/(m/s), (N*m)/(rad/s)]

# Adaptive gains, filter window, history stack size, and initial estimate
kg = 0.001 * np.ones(13)
ku = 50 * np.ones(13)
filter_window = 0.5  # s
history_size = 80
adapt0 = [512.5, 625, 0, 1000, -3.125, -31.25, -50, 0, 0, -31.25, -7.8125, 0, -81.25]

# Path to track
path_type = 'figure8'  # 'waypoint', 'sequence', 'circle', or 'figure8'
target = [10, 20, 0] # [m(x), m(y), rad] or [m(r), s(T), -]
vmax = [2, 1, 2]  # [m/s, m/s, rad/s]
amax = [1, 1, 1] # [m/s^2, m/s^2, rad/s^2]

# Rate at which controller is called
dt_c = dt  # s

# Initialize controller
controller = Controller(dt_c, q, target, path_type, kp, kd, kg, ku, umax, vmax, amax, history_size, filter_window, adapt0)

################################################# EQUATIONS OF MOTION

def dynamics(q, u):
	"""
	Returns state derivative (qdot).
	Takes control input (u) and current state (q).

	"""
	# Externally set parameters
	global m, Iz, xg, wm_xu, wm_yv, wm_yr, wm_nr,\
	       d_xuu, d_yvv, d_nrr, d_yrr, d_yrv, d_yvr,\
	       d_nvv, d_nrv, d_nvr

	# Mass matrix
	M = np.array([
	              [m - wm_xu,            0,            0],
	              [        0,    m - wm_yv, m*xg - wm_yr],
	              [        0, m*xg - wm_yr,   Iz - wm_nr]
	            ])

	# Centripetal coriolis matrix
	C = np.array([
	              [                                     0,                0, (wm_yr - m*xg)*q[5] + (wm_yv - m)*q[4]],
	              [                                     0,                0,                       (m - wm_xu)*q[3]],
	              [(m*xg - wm_yr)*q[5] + (m - wm_yv)*q[4], (wm_xu - m)*q[3],                                      0]
	            ])

	# Drag matrix
	D = np.array([
	              [-d_xuu*abs(q[3]),                                    0,                                    0],
	              [               0, -(d_yvv*abs(q[4]) + d_yrv*abs(q[5])), -(d_yvr*abs(q[4]) + d_yrr*abs(q[5]))],
	              [               0, -(d_nvv*abs(q[4]) + d_nrv*abs(q[5])), -(d_nvr*abs(q[4]) + d_nrr*abs(q[5]))]
	            ])

	# Rotation matrix (orientation, converts body to world)
	R = np.array([
	              [np.cos(q[2]), -np.sin(q[2]), 0],
	              [np.sin(q[2]),  np.cos(q[2]), 0],
	              [           0,             0, 1]
	            ])

	# Actuator saturation
	for i, mag in enumerate(abs(u)):
		if mag > umax[i]:
			u[i] = umax[i] * np.sign(u[i])

	# M*vdot + C*v + D*v = u  and  etadot = R*v
	return np.concatenate((R.dot(q[3:]), np.linalg.inv(M).dot(u - (C + D).dot(q[3:]))))

################################################# SIMULATION

# Define time domain
t_arr = np.arange(0, T, dt)

# Preallocate results memory
q_history = np.zeros((len(t_arr), len(q)))
qref_history = np.zeros((len(t_arr), len(q)))
u_history = np.zeros((len(t_arr), len(target)))
adapt_history = np.zeros((len(t_arr), len(adapt0)))
target_history = np.zeros((len(t_arr), len(target)))
aref_history = np.zeros((len(t_arr), len(target)))
u = np.zeros(len(target))
update_count = 0

# Integrate dynamics using first-order forward stepping
for i, t in enumerate(t_arr):

	# Controller's decision
	if t > update_count * dt_c:
		sensor_noise = noise_mean + noise_stdv*np.random.randn(len(q))
		u = controller.get_effort(q + sensor_noise, dt_c)
		update_count += 1

	# Record this instant
	u_history[i, :] = u
	q_history[i, :] = q
	qref_history[i, :] = controller.qref
	adapt_history[i, :] = controller.adapt
	target_history[i, :] = controller.target
	aref_history[i, :] = controller.aref

	# Quit early if something breaks
	if controller.kill:
		ignore = int(0.1 / dt)
		if i > ignore:
			u_history[(i-ignore):, :] = u_history[i-ignore, :]
			q_history[(i-ignore):, :] = q_history[i-ignore, :]
			qref_history[(i-ignore):, :] = qref_history[i-ignore, :]
			adapt_history[(i-ignore):, :] = adapt_history[i-ignore, :]
			target_history[(i-ignore):, :] = target_history[i-ignore, :]
			aref_history[(i-ignore):, :] = aref_history[i-ignore, :]
			t_arr[(i-ignore):] = t_arr[i]
		break

	# Modify any time-varying parameters
	pass

	# Step forward, qnext = qlast + qdot*dt
	q = q + dynamics(q, u)*dt

	# Wrap heading angle
	q[2] = np.arctan2(np.sin(q[2]), np.cos(q[2]))

################################################# VISUALIZATION

# Figure for individual results
fig1 = plt.figure()
fig1.suptitle('State Evolution', fontsize=20)
fig1rows = 2
fig1cols = 4

# Plot x position
ax1 = fig1.add_subplot(fig1rows, fig1cols, 1)
ax1.set_title('X Position (m)', fontsize=16)
ax1.plot(t_arr, q_history[:, 0], 'k',
		 t_arr, qref_history[:, 0], 'g--')
ax1.grid(True)

# Plot y position
ax1 = fig1.add_subplot(fig1rows, fig1cols, 2)
ax1.set_title('Y Position (m)', fontsize=16)
ax1.plot(t_arr, q_history[:, 1], 'k',
		 t_arr, qref_history[:, 1], 'g--')
ax1.grid(True)

# Plot yaw position
ax1 = fig1.add_subplot(fig1rows, fig1cols, 3)
ax1.set_title('Heading (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 2]), 'k',
		 t_arr, np.rad2deg(qref_history[:, 2]), 'g--')
ax1.grid(True)

# Plot control efforts
ax1 = fig1.add_subplot(fig1rows, fig1cols, 4)
ax1.set_title('Wrench (N, N, N*m)', fontsize=16)
ax1.plot(t_arr, u_history[:, 0], 'b',
		 t_arr, u_history[:, 1], 'g',
		 t_arr, u_history[:, 2], 'r')
ax1.grid(True)

# Plot x velocity
ax1 = fig1.add_subplot(fig1rows, fig1cols, 5)
ax1.set_title('Surge (m/s)', fontsize=16)
ax1.plot(t_arr, q_history[:, 3], 'k',
		 t_arr, qref_history[:, 3], 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot y velocity
ax1 = fig1.add_subplot(fig1rows, fig1cols, 6)
ax1.set_title('Sway (m/s)', fontsize=16)
ax1.plot(t_arr, q_history[:, 4], 'k',
		 t_arr, qref_history[:, 4], 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot yaw velocity
ax1 = fig1.add_subplot(fig1rows, fig1cols, 7)
ax1.set_title('Yaw (deg/s)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(q_history[:, 5]), 'k',
		 t_arr, np.rad2deg(qref_history[:, 5]), 'g--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# Plot adaptive estimates
#                       [m11,       m22,         m23,        m33,       d11u, d22v,  d22r,  d23v,  d23r,  d32v,  d32r,   d33v, d33r]
#                         0          1            2           3           4     5      6     7       8      9     10      11    12
adapt_true = np.array([m - wm_xu, m - wm_yv, m*xg - wm_yr, Iz - wm_nr, d_xuu, d_yvv, d_yrv, d_yvr, d_yrr, d_nvv, d_nrv, d_nvr, d_nrr])
ax1 = fig1.add_subplot(fig1rows, fig1cols, 8)
ax1.set_title('Adaptive Estimates', fontsize=16)
colors=plt.cm.rainbow(np.linspace(0, 1, 13))
for i in xrange(len(adapt_true)):
	ax1.plot(t_arr, adapt_history[:, i], color=colors[i])
	ax1.plot([0, t_arr[-1]], adapt_true[i]*np.ones(2), color=colors[i], linestyle='--')
ax1.set_xlabel('Time (s)')
ax1.grid(True)

# # Plot aref x
# ax1 = fig1.add_subplot(fig1rows, fig1cols, 9)
# ax1.set_title('aref_X (m/s^2)', fontsize=16)
# ax1.plot(t_arr, aref_history[:, 0], 'r--')
# ax1.grid(True)

# # Plot aref y
# ax1 = fig1.add_subplot(fig1rows, fig1cols, 10)
# ax1.set_title('aref_Y (m/s^2)', fontsize=16)
# ax1.plot(t_arr, aref_history[:, 1], 'r--')
# ax1.grid(True)

# # Plot aref heading
# ax1 = fig1.add_subplot(fig1rows, fig1cols, 11)
# ax1.set_title('aref_yaw (rad/s^2)', fontsize=16)
# ax1.plot(t_arr, aref_history[:, 2], 'r--')
# ax1.grid(True)

YY_sum = np.zeros((13, 13))
for i, pair in enumerate(controller.history_stack):
	YY_sum = YY_sum + pair['Yi'].T.dot(pair['Yi'])
vals, vecs = np.linalg.eig(YY_sum)
print('\nFinal adaptation error: {}\n'.format(np.round(adapt_true - controller.adapt, 1)))
print('Final eigenvalues:\n{}\n'.format(np.real(vals)))


# Create plot for in-depth look at adaptation
pnames = ["m-wm_xu", "m-wm_yv", "m*xg-wm_yr", "Iz-wm_nr",
          "d_xuu", "d_yvv", "d_nrr", "d_yrr", "d_yrv",
          "d_yvr", "d_nvv", "d_nrv", "d_nvr"]
ttild = np.array([adapt_true]) - adapt_history
ttild_norm = np.zeros(len(t_arr))
fig1a = plt.figure()
fig1a.suptitle('Adaptation', fontsize=20)
ax1a = fig1a.add_subplot(3, 1, 2)
ax1a.set_ylabel('Errors', fontsize=16)
for i in xrange(len(adapt_true)):
	ax1a.plot(t_arr, ttild[:, i], color=colors[i])
ax1a.grid(True)
ax1a = fig1a.add_subplot(3, 1, 3)
ax1a.set_ylabel('Error Norm', fontsize=16)
ax1a.set_xlabel('Time (s)', fontsize=16)
for i in xrange(len(t_arr)):
	ttild_norm[i] = np.linalg.norm(ttild[i, :])
ax1a.plot(t_arr, ttild_norm)
ax1a.grid(True)
ax1a = fig1a.add_subplot(3, 1, 1)
ax1a.set_ylabel('Estimates', fontsize=16)
for i in xrange(len(adapt_true)):
    ax1a.plot(t_arr, adapt_history[:, i], color=colors[i])
    ax1a.plot([0, t_arr[-1]], adapt_true[i]*np.ones(2), color=colors[i], label=pnames[i], linestyle='--')
ax1a.legend(loc='upper right')
ax1a.grid(True)


# # Create figure for parametric results
# fig2 = plt.figure()
# fig2.suptitle('Pose Space', fontsize=20)
# ax2 = fig2.add_subplot(1, 1, 1)
# ax2.set_xlabel('X (m)')
# ax2.set_ylabel('Y (m)')

# # Plot parametric
# ax2.plot(q_history[:, 0], q_history[:, 1], 'k', qref_history[:, 0], qref_history[:, 1], 'g--')
# ax2.scatter(q_history[0, 0], q_history[0, 1], color='r', s=50)
# if path_type in ['waypoint', 'sequence']:
# 	ax2.scatter(target_history[:, 0], target_history[:, 1], color='g', s=50)
# ax2.grid(True)


# Figure for animation
fig3 = plt.figure()
fig3.suptitle('Evolution')
ax3 = fig3.add_subplot(1, 1, 1)
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_aspect('equal', 'datalim')
ax3.grid(True)

# Points and lines for representing positions and headings
pthick = 100
lthick = 3
llen = 1
p = ax3.scatter(q_history[0, 0], q_history[0, 1], color='k', s=pthick)
h = ax3.plot([q_history[0, 0], q_history[0, 0] + llen*np.cos(q_history[0, 2])],
             [q_history[0, 1], q_history[0, 1] + llen*np.sin(q_history[0, 2])], color='k', linewidth=lthick)
pref = ax3.scatter(qref_history[0, 0], qref_history[0, 1], color='b', s=pthick)
href = ax3.plot([qref_history[0, 0], qref_history[0, 0] + llen*np.cos(qref_history[0, 2])],
                [qref_history[0, 1], qref_history[0, 1] + llen*np.sin(qref_history[0, 2])], color='b', linewidth=lthick)
if path_type in ['waypoint', 'sequence']:
	ptar = ax3.scatter(target_history[0, 0], target_history[0, 1], color='g', s=pthick)
	htar = ax3.plot([target_history[0, 0], target_history[0, 0] + llen*np.cos(target_history[0, 2])],
                [target_history[0, 1], target_history[0, 1] + llen*np.sin(target_history[0, 2])], color='g', linewidth=lthick)

# Plot entirety of actual trajectory
if outline_path:
	ax3.plot(q_history[:, 0], q_history[:, 1], 'k--', qref_history[:, 0], qref_history[:, 1], 'g--')

# Function for updating the animation frame
def update_ani(arg, ii=[0]):

	i = ii[0]  # don't ask...

	if np.isclose(t_arr[i], np.around(t_arr[i], 1)):
		fig3.suptitle('Evolution (Time: {})'.format(t_arr[i]), fontsize=24)

	p.set_offsets((q_history[i, 0], q_history[i, 1]))
	h[0].set_data([q_history[i, 0], q_history[i, 0] + llen*np.cos(q_history[i, 2])],
	              [q_history[i, 1], q_history[i, 1] + llen*np.sin(q_history[i, 2])])
	pref.set_offsets((qref_history[i, 0], qref_history[i, 1]))
	href[0].set_data([qref_history[i, 0], qref_history[i, 0] + llen*np.cos(qref_history[i, 2])],
	                 [qref_history[i, 1], qref_history[i, 1] + llen*np.sin(qref_history[i, 2])])
	if path_type in ['waypoint', 'sequence']:
		ptar.set_offsets((target_history[i, 0], target_history[i, 1]))
		htar[0].set_data([target_history[i, 0], target_history[i, 0] + llen*np.cos(target_history[i, 2])],
	                 [target_history[i, 1], target_history[i, 1] + llen*np.sin(target_history[i, 2])])

	ii[0] += int(1 / (dt * framerate))
	if ii[0] >= len(t_arr):
		print("Resetting animation!")
		ii[0] = 0

	if path_type in ['waypoint', 'sequence']:
		return [p, h, pref, href, ptar, htar]
	else:
		return [p, h, pref, href]

# Run animation
ani = animation.FuncAnimation(fig3, func=update_ani, interval=dt*1000/speedup)
plt.show()

# Store data
if store_data:

	from scipy.io import savemat

	filename = 'data_{}_{}s'.format(path_type, T)

	data = {'time': t_arr,
	        'north': q_history[:, 0],
	        'east': q_history[:, 1],
	        'heading': q_history[:, 2],
	        'surge': q_history[:, 3],
	        'sway': q_history[:, 4],
	        'yawrate': q_history[:, 5],
	        'north_des': qref_history[:, 0],
	        'east_des': qref_history[:, 1],
	        'heading_des': qref_history[:, 2],
	        'surge_des': qref_history[:, 3],
	        'sway_des': qref_history[:, 4],
	        'yawrate_des': qref_history[:, 5],
	        'surge_force': u_history[:, 0],
	        'sway_force': u_history[:, 1],
	        'yaw_torque': u_history[:, 2],
	        'theta_hat': adapt_history,
	        'theta': adapt_true
	       }

	savemat(filename, data)

	print('Data saved!\n')
