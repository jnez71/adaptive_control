"""
# (todo)
#<<< finish documentation

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl
from collections import deque
from cmath import sqrt

################################################# PRIMARY CLASS

class Controller:

	def __init__(self, dt, q0, target, path_type,
				 kp, kd, kg, ku, kf,
				 umax, vmax, amax,
				 history_size, filter_window, adapt0):
		"""
		Set-up. Takes call-period, initial state, target pose, path type,
		gains, integral/filter window size, effort limit, maximum speed
		and acceleration, history stack size, selection type, and initial condition.

		"""
		self.ncontrols = len(umax)
		self.nparams = len(adapt0)
		if filter_window and np.isfinite(filter_window):
			self.nfilt = int(filter_window / dt)
		else:
			self.nfilt = 0

		self.set_gains(kp, kd, kg, ku, kf)
		self.set_limits(umax, vmax, amax)

		self.adapt = adapt0
		self.adapt_err = np.zeros(self.nparams)
		self.Lest = np.zeros(2)
		self.mest = np.zeros(2)
		self.gest = 0

		self.uf = np.zeros(self.ncontrols)
		self.Yuf = np.zeros((self.ncontrols, self.nparams))
		self.Yuf1 = np.zeros((self.ncontrols, self.nparams))

		self.uf_stack = deque([self.uf] * self.nfilt)
		self.Yuf1_stack = deque([self.Yuf1] * self.nfilt)

		self.q0 = q0
		self.q_stack = deque([q0] * self.nfilt)

		self.history_stack = deque([self.make_history_pair(self.Yuf, self.uf)] * history_size)
		self.history_size = history_size
		self.history_eig = 0

		self.YY_stack = deque([np.zeros((self.nparams, self.nparams))] * history_size)
		self.YY_sum = np.zeros((self.nparams, self.nparams))

		self.time = 0
		self.set_path(q0, target, path_type, dt)

		self.rep = np.zeros(self.ncontrols)
		self.rep_T = np.zeros(self.ncontrols)
		self.rep_stack = deque([self.rep] * self.ncycle)

		self.kill = False

########################

	def set_gains(self, kp, kd, kg, ku, kf):
		"""
		Sets proportional, derivative, adaptive, and filter gains.

		"""
		self.kp = np.array(kp, dtype=np.float32)
		self.kd = np.array(kd, dtype=np.float32)
		self.kr = self.kp / self.kd

		if type(kg) is str:
			if kg == 'LS':
				self.kg = 100*np.eye(self.nparams)
				self.use_LS = True
			else:
				raise ValueError("Did you mean kg = 'LS' (least squares)?")
		else:
			self.kg = np.diag(kg)
			self.use_LS = False

		self.ku = np.diag(ku)
		self.kf = np.array(kf, dtype=np.float32)

########################

	def set_limits(self, umax, vmax, amax):
		"""
		Sets model limits.
		Uses the limits to compute a model reference for tracking,
		and uses repmax for limiting repetative learning.

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

		self.repmax = np.array([15, 15])

########################

	def set_path(self, q0, target, path_type, dt):
		"""
		Resets controller time and reference acceleration.
		Sets the path initial state, the target position, and the
		type of path. Updates reference q to its initial t=0 value.
		If the path will be cyclic, repetative learning is enabled.
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

		if path_type == 'cycle':
			self.use_RL = True
		else:
			self.use_RL = False

		self.Tcycle = 5  # s
		self.ncycle = int(2 * self.Tcycle / dt)

		self.update_ref(0)

########################

	def get_effort(self, q, dt):
		"""
		#<<< explain adaptive controller

		"""
		# Tracking errors
		E = self.qref[:2] - q[:2]
		Edot = self.qref[2:] - q[2:]
		tracking_err = self.kr*E + Edot

		# Tracking regressor
		Y = np.array([
		              [np.cos(q[0]),
		               self.aref[0] - self.kr[0]*q[2] + self.kr[0]*self.qref[2],
		               np.cos(q[0] + q[1]),
		               np.cos(q[1])*(2*self.aref[0] + self.aref[1] - 2*self.kr[0]*q[2] - self.kr[1]*q[3] + 2*self.kr[0]*self.qref[2] + self.kr[1]*self.qref[3]) - q[3]*np.sin(q[1])*(2*q[2] + q[3]),
		               self.aref[0] + self.aref[1] - self.kr[0]*q[2] - self.kr[1]*q[3] + self.kr[0]*self.qref[2] + self.kr[1]*self.qref[3]],
		              [0,
		               0,
		               np.cos(q[0] + q[1]),
		               q[2]**2*np.sin(q[1]) + np.cos(q[1])*(self.aref[0] - self.kr[0]*q[2] + self.kr[0]*self.qref[2]),
		               self.aref[0] + self.aref[1] - self.kr[0]*q[2] - self.kr[1]*q[3] + self.kr[0]*self.qref[2] + self.kr[1]*self.qref[3]]
		            ])

		# Control law
		u = self.kp*E + self.kd*Edot + Y.dot(self.adapt) + self.rep

		# Learning gradient gain
		if self.use_LS:
			# Approximate least-squares gain choice
			self.kg = self.kg - (self.kg.dot(self.ku.dot(self.Yuf.T.dot(self.Yuf))).dot(self.kg))*dt

		# Update adaptation
		self.adapt = self.adapt + self.kg.dot(Y.T.dot(tracking_err) + self.ku.dot(self.adapt_err))*dt
		if self.use_RL:
			self.rep = np.clip(self.rep_T, -self.repmax, self.repmax) + self.kd*tracking_err
			self.rep_stack.append(self.rep)
			self.rep_T = self.rep_stack.popleft()

		# Update filtered prediction regressor, filtered control effort, and learning history stack
		self.update_learning(q, u, dt)

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

	def update_learning(self, q, u, dt):
		"""
		#<<< explain concurrent learning

		"""
		# Instantaneous parts of filtered prediction regressor
		Yuf2_now = np.array([
		                     [0, q[2], 0, np.cos(q[1])*(2*q[2] + q[3]), q[2] + q[3]],
		                     [0, 0, 0, q[2]*np.cos(q[1]), q[2] + q[3]]
		                   ])

		Yuf2_then = np.array([
                      [0, self.q0[2], 0, np.cos(self.q0[1])*(2*self.q0[2] + self.q0[3]), self.q0[2] + self.q0[3]],
                      [0, 0, 0, self.q0[2]*np.cos(self.q0[1]), self.q0[2] + self.q0[3]]
                    ])

		Yuf2 = Yuf2_now - Yuf2_then

		# Convolutional filtering of prediction regressor and control effort...
		if self.kf:
			self.Yuf = self.kf*(self.Yuf1 + Yuf2)
			Yuf1dot = np.array([
			                    [np.cos(q[0]), -self.kf*q[2], np.cos(q[0] + q[1]), -self.kf*np.cos(q[1])*(2*q[2] + q[3]), -self.kf*(q[2] + q[3])],
			                    [0, 0, np.cos(q[0] + q[1]), q[2]*((q[2] + q[3])*np.sin(q[1]) - self.kf*np.cos(q[1])), -self.kf*(q[2] + q[3])]
			                  ])
			# infinite window continuous sum...
			if not self.nfilt:
				self.uf = self.uf + self.kf*(u - self.uf)*dt
				self.Yuf1 = self.Yuf1 + (Yuf1dot - self.kf*self.Yuf1)*dt
			# ...or finite window push pop
			else:
				self.uf_stack.append(self.kf*(u - self.uf)*dt)
				self.uf = (self.uf - self.uf_stack.popleft()) + self.uf_stack[-1]
				self.Yuf1_stack.append((Yuf1dot - self.kf*self.Yuf1)*dt)
				self.Yuf1 = (self.Yuf1 - self.Yuf1_stack.popleft()) + self.Yuf1_stack[-1]
				self.q_stack.append(q)
				self.q0 = self.q_stack.popleft()

		# ...or integral filtering of prediction regressor and control effort if kf = 0
		else:
			self.Yuf = self.Yuf1 + Yuf2
			Yuf1dot = np.array([
			                    [np.cos(q[0]), 0, np.cos(q[0] + q[1]), 0, 0],
			                    [0, 0, np.cos(q[0] + q[1]), q[2]*(q[2] + q[3])*np.sin(q[1]), 0]
			                  ])
			# infinite window continuous sum...
			if not self.nfilt:
				self.uf = self.uf + u*dt
				self.Yuf1 = self.Yuf1 + Yuf1dot*dt
			# ...or finite window push pop
			else:
				self.uf_stack.append(u*dt)
				self.uf = (self.uf - self.uf_stack.popleft()) + self.uf_stack[-1]
				self.Yuf1_stack.append(Yuf1dot*dt)
				self.Yuf1 = (self.Yuf1 - self.Yuf1_stack.popleft()) + self.Yuf1_stack[-1]
				self.q_stack.append(q)
				self.q0 = self.q_stack.popleft()

		# If stack size is > 0 then use selective learning...
		if self.history_size:

			# Candidate data point
			new_data = self.make_history_pair(self.Yuf, self.uf)
			new_YY = self.Yuf.T.dot(self.Yuf)

			# If buffer is full...
			if self.time > dt*self.history_size:

				# Space for storing minimum eigenvalues during new data point testing
				eig_mins = np.zeros(self.history_size)

				# YY_sum if we add new data but don't remove any
				extended_sum = self.YY_sum + new_YY

				# Test all possible insertions of the new data
				for i in xrange(self.history_size):
					candidate_sum = extended_sum - self.YY_stack[i]
					try:
						assert np.isfinite(candidate_sum[0, 0])
						eig_mins[i] = npl.eigvalsh(candidate_sum)[0]
					except (npl.LinAlgError, AssertionError):
						print("ADAPTATION UNSTABLE: try a smaller kg (or pick kg='LS'), or try a smaller stack_size.")
						self.kill = True
						return 0

				# Take best possible insertion if it raises the minimum eigenvalue of our current stack
				hotseat = np.argmax(eig_mins)
				if eig_mins[hotseat] > self.history_eig and not self.saturated:
					# Print if wisdom has increased significantly
					if eig_mins[hotseat] - self.history_eig > 0.001:
						print('Significant: {}  @ time: {}'.format(np.round(self.history_eig*100, 1), self.time))
					# Update history
					self.history_stack[hotseat] = new_data
					self.history_eig = eig_mins[hotseat]
					self.YY_sum = extended_sum - self.YY_stack[hotseat]
					self.YY_stack[hotseat] = new_YY

			# ...until then just learn regardless
			else:
				self.history_stack.append(new_data)
				self.history_stack.popleft()					
				self.YY_stack.append(new_YY)
				self.YY_sum = (self.YY_sum - self.YY_stack.popleft()) + new_YY
				print('Buffering @ time: {}'.format(self.time))

			# Update estimated adaptation error
			self.adapt_err = np.zeros(self.nparams)
			for i, pair in enumerate(self.history_stack):
				self.adapt_err = self.adapt_err + pair['Yi'].T.dot(pair['ui'] - pair['Yi'].dot(self.adapt))
		
		# ...otherwise just use newest data point ("composite adaptation")
		else:
			self.adapt_err = self.Yuf.T.dot(self.uf - self.Yuf.dot(self.adapt))

		# Solve for system parameters using dynamic parameter estimates, taking a great guess at g
		if all(np.around(abs(self.adapt), 2)):
			self.Lest = 9.81 * abs(np.array([self.adapt[1] / self.adapt[0], self.adapt[4] / self.adapt[2]]))
			self.mest[1] = abs(self.adapt[4] / self.Lest[1]**2)
			self.mest[0] = abs((self.adapt[1] / self.Lest[0]**2) - self.mest[1])

########################
	
	def make_history_pair(self, Yi, ui):
		"""
		Creates a history pair as a dictionary containing keys 'Yi' and 'ui',
		which are the filtered regressor and filtered effort for that instant.

		"""
		return {'Yi': Yi, 'ui': ui}

########################

	def update_ref(self, dt):
		"""
		#<<< explain trajgen

		"""
		self.path_time = self.path_time + dt

		if self.path_type == 'train':
			Eref = self.target[:2] - self.qref[:2]
			Erefdot = -self.qref[2:]
			uref = self.kp*Eref + self.kd*Erefdot
			self.qref = self.qref + self.reference_dynamics(self.qref, uref)*dt
			if self.path_time > self.Tcycle:
				self.set_path(self.qref, 2*np.pi*(np.random.rand(2) - 0.5), 'train', dt)

		elif self.path_type in ['waypoint', 'random']:
			target_q = self.kinem_reverse(np.concatenate((self.target, [0, 0])), self.qref)[:2]
			Eref = target_q[:2] - self.qref[:2]
			Erefdot = -self.qref[2:]
			uref = self.kp*Eref + self.kd*Erefdot
			self.qref = self.qref + self.reference_dynamics(self.qref, uref)*dt

			if self.path_type == 'random' and self.path_time > self.Tcycle:
				searching = True
				while searching:
					target = sum(self.Lest)*(np.random.rand(2) - 0.5)
					if (all(np.around(abs(self.Lest), 5)) and
                        abs((npl.norm(target)**2 - self.Lest[0]**2 - self.Lest[1]**2) / (2*self.Lest[0]*self.Lest[1])) <= 1 and
                        npl.norm(target - self.target) > 1):
						searching = False
				self.set_path(self.qref, target, 'random', dt)

		elif self.path_type == 'cycle':
			Eref = self.target[:2] - self.qref[:2]
			Erefdot = -self.qref[2:]
			uref = self.kp*Eref + self.kd*Erefdot
			self.qref = self.qref + self.reference_dynamics(self.qref, uref)*dt
			if self.path_time > self.Tcycle:
				self.set_path(self.qref, -self.target, 'cycle', dt)

		else:
			raise ValueError("Invalid path_type.")

########################

	def reference_dynamics(self, qref, uref):
		"""
		Computes reference state derivative (qrefdot).
		Takes reference state (qref) and reference control input (uref).

		"""
		# Imposed actuator saturation
		for i, mag in enumerate(abs(uref)):
			if mag > self.umaxref[i]:
				uref[i] = self.umaxref[i] * np.sign(uref[i])

		# Simple linear evolution
		return np.concatenate((qref[2:] , (uref - self.dref*qref[2:]) / self.mref))

########################

	def kinem_reverse(self, x, qlast=None):
		"""
		Given some end effector state x, solves for the corresponding joint state q.
		Optionally uses the last joint state qlast to decide on the closest new q solution.

		"""
		if all(np.around(abs(self.Lest), 5)):
			c2 = (npl.norm(x[:2])**2 - self.Lest[0]**2 - self.Lest[1]**2) / (2*self.Lest[0]*self.Lest[1])
		else:
			c2 = (npl.norm(x[:2])**2 - 2) / 2

		s2a = np.real(sqrt(1 - c2**2))
		s2b = -s2a

		Jp = np.array([[self.Lest[0] + self.Lest[1]*c2, -self.Lest[1]*s2a],
                       [self.Lest[1]*s2a, self.Lest[0] + self.Lest[1]*c2]
                     ])

		if abs(c2) > 1 or np.isclose(npl.det(Jp), 0):
			ta = 2*np.pi*(np.random.rand(2)-0.5)
			tb = 2*np.pi*(np.random.rand(2)-0.5)

		else:
			c1a, s1a = npl.inv(Jp).dot(x[:2])
			c1b, s1b = npl.inv(Jp.T).dot(x[:2])
			ta = np.array([np.arctan2(s1a, c1a), np.arctan2(s2a, c2)])
			tb = np.array([np.arctan2(s1b, c1b), np.arctan2(s2b, c2)])

		if qlast is None or npl.norm(ta-qlast[:2]) < npl.norm(tb-qlast[:2]):
			t = ta
		else:
			t = tb

		Jv = np.array([[-(self.Lest[0]*np.sin(t[0]) + self.Lest[1]*np.sin(t[0]+t[1])), -self.Lest[1]*np.sin(t[0]+t[1])],
                       [self.Lest[0]*np.cos(t[0]) + self.Lest[1]*np.cos(t[0]+t[1]), self.Lest[1]*np.cos(t[0]+t[1])]
                     ])

		if np.isclose(npl.det(Jv), 0):
			w = np.zeros(2)
		else:
			w = npl.inv(Jv).dot(x[2:])
		
		return np.concatenate((t, w))
