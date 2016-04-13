"""
Quickly threw together an NN controller for a project.
Carries over from parametric CL controller implementations.
Needs a lot of cleanup.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl

################################################# PRIMARY CLASS

class NN_controller:

	def __init__(self, dt, q0, target, path_type,
				 kp, kd, n, kv, kw,
				 umax, vmax, amax):
		"""
		document

		"""
		self.nstates = len(q0)
		self.ncontrols = len(umax)
		self.nsigs = n

		self.sig = lambda x: np.concatenate(([1], np.tanh(x)))
		self.sigp = lambda x: np.tile(1/(np.cosh(x)**2), (self.nsigs+1, 1))

		self.set_gains(kp, kd, kv, kw)
		self.set_limits(umax, vmax, amax)

		self.V = np.zeros((self.nstates+1, self.nsigs))
		self.W = np.zeros((self.nsigs+1, self.ncontrols))
		self.y = np.zeros(self.ncontrols)

		self.time = 0
		self.set_path(q0, target, path_type, dt)
		self.kill = False

########################

	def set_gains(self, kp, kd, kv, kw):
		"""
		document

		"""
		self.kp = np.array(kp, dtype=np.float32)
		self.kd = np.array(kd, dtype=np.float32)
		self.kr = self.kp / self.kd

		self.kv = np.array(kv, dtype=np.float32)
		self.kw = np.array(kw, dtype=np.float32)

########################

	def set_limits(self, umax, vmax, amax):
		"""
		Sets model limits.
		Uses the limits to compute a model reference for tracking,
		and uses repmax for limiting repetitive learning.

		"""
		self.umax = np.array(umax, dtype=np.float32)
		self.vmax = np.array(vmax, dtype=np.float32)
		self.amax = np.array(amax, dtype=np.float32)
		self.saturated = False

		if np.inf in self.umax or 0 in self.umax:
			self.umaxref = np.array([250, 30], dtype=np.float32)
		else:
			self.umaxref = self.umax

		self.dref = self.umaxref / self.vmax

		if np.inf in self.amax:
			self.mref = np.array([0.01, 0.01], dtype=np.float32)
		else:
			self.mref = self.umaxref / self.amax

########################

	def set_path(self, q0, target, path_type, dt):
		"""
		Resets controller time and reference acceleration.
		Sets the path initial state, the target position, and the
		type of path. Updates reference q to its initial t=0 value.
		If the path will be cyclic, repetitive learning is enabled.
		The path cycle period is hardcoded in.

		"""
		self.path_time = 0
		self.qref = np.array(q0)
		self.aref = np.zeros(self.ncontrols)
		self.path_type = path_type

		if path_type == 'train':
			self.target = 2*np.pi*(np.random.rand(2) - 0.5)
		else:
			self.target = np.array(target)

		self.update_ref(0)

########################

	def get_effort(self, q, dt):
		"""
		Returns the vector of torques as a PD controller plus
		a feedforward term that uses an estimate of the system's
		dynamics. The output is saturated at umax as
		specified by the user previously. Before returning the
		torques, the latest dynamics estimate is also updated.

		"""
		# Tracking errors
		E = self.qref[:2] - q[:2]
		Edot = self.qref[2:] - q[2:]
		r = self.kr*E + Edot

		# Control law
		u = self.kp*E + self.kd*Edot + self.y

		# Adapt NN
		if not self.saturated:
			x = np.concatenate(([1], q))
			VTx = self.V.T.dot(x)
			Wdot = self.kw.dot(np.outer(self.sig(VTx), r))
			Vdot = self.kv.dot(np.outer(x, r).dot(self.W.T).dot(self.sigp(VTx)))
			self.W = self.W + Wdot*dt
			self.V = self.V + Vdot*dt
			self.y = self.W.T.dot(self.sig(self.V.T.dot(x)))

		# Update reference trajectory and controller life time
		self.update_ref(dt)
		self.time = self.time + dt

		# Safety saturation of output
		self.saturated = False
		for i, mag in enumerate(abs(u)):
			if mag > self.umax[i]:
				u[i] = self.umax[i] * np.sign(u[i])
				self.saturated = True

		# Return effort torques
		return u

########################

	def update_ref(self, dt):
		"""
		Updates the reference state qref depending on the 
		settings created in set_path. In every case, a 
		spring-damper tuned to vmax and amax is used to 
		generate the profile between each discontinuous target.

		'train': sequence of random joint-space configurations

		"""
		self.path_time = self.path_time + dt

		if self.path_type == 'train':
			Eref = self.target[:2] - self.qref[:2]
			Erefdot = -self.qref[2:]
			uref = self.kp*Eref + self.kd*Erefdot
			self.qref = self.qref + self.reference_dynamics(self.qref, uref)*dt
			if self.path_time > 5:
				self.set_path(self.qref, 2*np.pi*(np.random.rand(2) - 0.5), 'train', dt)

		else:
			raise ValueError("Invalid path_type.")

########################

	def reference_dynamics(self, qref, uref):
		"""
		Computes reference state derivative (qrefdot).
		Takes reference state (qref) and reference control input (uref).
		Spring-damper model tuned to vmax (terminal velocity) and amax (saturation).

		"""
		# Imposed actuator saturation
		for i, mag in enumerate(abs(uref)):
			if mag > self.umaxref[i]:
				uref[i] = self.umaxref[i] * np.sign(uref[i])

		# Simple linear evolution
		return np.concatenate((qref[2:] , (uref - self.dref*qref[2:]) / self.mref))
