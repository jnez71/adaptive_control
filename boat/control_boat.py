"""
Concurrent-learning controller derived for
Fossen's dynamic model of a boat.

See:
<paper>

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl
from collections import deque

################################################# PRIMARY CLASS

class Controller:

	def __init__(self, dt, q0, target, path_type,
				 kp, kd, kg, ku,
				 umax, vmax, amax,
				 history_size, filter_window, adapt0):
		"""
		Set-up. Takes call-period, initial state, target pose, path type,
		feedback gains, learning gains, effort limit, speed and acceleration
		limits, history stack size, filter window time, and initial adaptation.

		"""
		self.ncontrols = len(umax)
		self.nparams = len(adapt0)
		if filter_window and np.isfinite(filter_window):
			self.nfilt = int(filter_window / dt)
		else:
			self.nfilt = 0

		self.set_gains(kp, kd, kg, ku)
		self.set_limits(umax, vmax, amax)

		self.adapt = np.array(adapt0, dtype=np.float32)
		self.adapt_err = np.zeros(self.nparams)
		self.last_adaptdot = np.zeros(self.nparams)

		self.uf = np.zeros(self.ncontrols)
		self.Yuf = np.zeros((self.ncontrols, self.nparams))
		self.Yuf1 = np.zeros((self.ncontrols, self.nparams))

		self.uf_stack = deque([self.uf] * self.nfilt)
		self.Yuf1_stack = deque([self.Yuf1] * self.nfilt)

		self.q0 = q0
		self.q_stack = deque([q0] * self.nfilt)

		self.last_Ru = np.zeros(self.ncontrols)
		self.last_Yuf1dot = np.zeros((self.ncontrols, self.nparams))

		self.history_stack = deque([self.make_history_pair(self.Yuf, self.uf)] * history_size)
		self.history_size = history_size
		self.history_eig = 0

		self.YY_stack = deque([np.zeros((self.nparams, self.nparams))] * history_size)
		self.YY_sum = np.zeros((self.nparams, self.nparams))

		self.time = 0
		self.set_path(q0, target, path_type)
		self.kill = False

########################

	def set_gains(self, kp, kd, kg, ku):
		"""
		Sets proportional, derivative, tracking, and learning gains.

		"""
		self.kp = np.array(kp, dtype=np.float32)
		self.kd = np.array(kd, dtype=np.float32)
		self.kr = (self.kp - 1) / kd

		self.ku = np.diag(ku)
		if type(kg) is str:
			if kg == 'LS':
				self.kg = 100*np.eye(self.nparams)
				self.use_LS = True
			else:
				raise ValueError("Did you mean kg = 'LS' (least squares)?")
		else:
			self.kg = np.diag(kg)
			self.use_LS = False

########################

	def set_limits(self, umax, vmax, amax):
		"""
		Sets model limits.
		Uses the limits to compute a model reference for tracking.

		"""
		self.umax = np.array(umax, dtype=np.float32)
		self.vmax = np.array(vmax, dtype=np.float32)
		self.amax = np.array(amax, dtype=np.float32)
		self.saturated = False

		if np.inf in self.umax or 0 in self.umax:
			self.umaxref = np.array([10000, 10000, 10000], dtype=np.float32)
		else:
			self.umaxref = self.umax

		self.dref = self.umaxref / self.vmax

		if np.inf in self.amax:
			self.mref = np.array([10, 10, 10], dtype=np.float32)
		else:
			self.mref = self.umaxref / self.amax

########################

	def set_path(self, q0, target, path_type):
		"""
		Resets controller time and reference acceleration.
		Sets the path initial state, the target position, and the
		type of path. Updates reference q to its initial t=0 value.

		"""
		self.path_time = 0
		self.qref = np.array(q0, dtype=np.float32)
		self.aref = np.zeros(self.ncontrols)
		
		self.path_type = path_type
		self.target = np.array(target, dtype=np.float32)
		self.update_ref(0)

########################

	def get_effort(self, q, dt):
		"""
		#<<< explain adaptive controller

		"""
		# Rotation matrix (orientation, converts body to world)
		c = np.cos(q[2])
		s = np.sin(q[2])
		R = np.array([
		              [c, -s, 0],
		              [s,  c, 0],
		              [0,  0, 1]
		            ])

		# Tracking errors
		Eh = self.get_heading_error(self.qref[2], R)
		E = np.concatenate((self.qref[:2] - q[:2], [Eh]))
		Edot_body = self.qref[3:] - q[3:]
		Edot = R.dot(Edot_body)
		r = Edot + self.kr*E

		# Tracking regressor
		Y = np.array([
		              [c**2*q[5]*(self.kr[1]*(self.qref[1] - q[1]) + np.cos(self.qref[2])*self.qref[4] + np.sin(self.qref[2])*self.qref[3]) - s*q[3]*(self.qref[5] + self.kr[2]*(Eh)) - c**2*(self.kr[0]*(c*q[3] - np.cos(self.qref[2])*self.qref[3] - s*q[4] + np.sin(self.qref[2])*self.qref[4]) - np.cos(self.qref[2])*self.aref[0] + np.sin(self.qref[2])*self.aref[1] + np.cos(self.qref[2])*self.qref[5]*self.qref[4] + self.qref[5]*np.sin(self.qref[2])*self.qref[3]) + c*s*(np.cos(self.qref[2])*self.aref[1] - self.kr[1]*(c*q[4] - np.cos(self.qref[2])*self.qref[4] + s*q[3] - np.sin(self.qref[2])*self.qref[3]) + np.sin(self.qref[2])*self.aref[0] + np.cos(self.qref[2])*self.qref[5]*self.qref[3] - self.qref[5]*np.sin(self.qref[2])*self.qref[4]) + c*q[5]*s*(self.kr[0]*(q[0] - self.qref[0]) - np.cos(self.qref[2])*self.qref[3] + np.sin(self.qref[2])*self.qref[4]), q[5]*s**2*(self.kr[1]*(self.qref[1] - q[1]) + np.cos(self.qref[2])*self.qref[4] + np.sin(self.qref[2])*self.qref[3]) - c*q[4]*(self.qref[5] + self.kr[2]*(Eh)) - s**2*(self.kr[0]*(c*q[3] - np.cos(self.qref[2])*self.qref[3] - s*q[4] + np.sin(self.qref[2])*self.qref[4]) - np.cos(self.qref[2])*self.aref[0] + np.sin(self.qref[2])*self.aref[1] + np.cos(self.qref[2])*self.qref[5]*self.qref[4] + self.qref[5]*np.sin(self.qref[2])*self.qref[3]) - c*s*(np.cos(self.qref[2])*self.aref[1] - self.kr[1]*(c*q[4] - np.cos(self.qref[2])*self.qref[4] + s*q[3] - np.sin(self.qref[2])*self.qref[3]) + np.sin(self.qref[2])*self.aref[0] + np.cos(self.qref[2])*self.qref[5]*self.qref[3] - self.qref[5]*np.sin(self.qref[2])*self.qref[4]) - c*q[5]*s*(self.kr[0]*(q[0] - self.qref[0]) - np.cos(self.qref[2])*self.qref[3] + np.sin(self.qref[2])*self.qref[4]), - s*(self.aref[2] - self.kr[2]*(q[5] - self.qref[5])) - c*q[5]*(self.qref[5] + self.kr[2]*(Eh)), 0, -c*q[3]*abs(q[3]),  s*q[4]*abs(q[4]),  s*q[4]*abs(q[5]),  q[5]*s*abs(q[4]),  q[5]*s*abs(q[5]), 0, 0, 0, 0],
		              [s**2*(np.cos(self.qref[2])*self.aref[1] - self.kr[1]*(c*q[4] - np.cos(self.qref[2])*self.qref[4] + s*q[3] - np.sin(self.qref[2])*self.qref[3]) + np.sin(self.qref[2])*self.aref[0] + np.cos(self.qref[2])*self.qref[5]*self.qref[3] - self.qref[5]*np.sin(self.qref[2])*self.qref[4]) + c*q[3]*(self.qref[5] + self.kr[2]*(Eh)) - c*s*(self.kr[0]*(c*q[3] - np.cos(self.qref[2])*self.qref[3] - s*q[4] + np.sin(self.qref[2])*self.qref[4]) - np.cos(self.qref[2])*self.aref[0] + np.sin(self.qref[2])*self.aref[1] + np.cos(self.qref[2])*self.qref[5]*self.qref[4] + self.qref[5]*np.sin(self.qref[2])*self.qref[3]) + q[5]*s**2*(self.kr[0]*(q[0] - self.qref[0]) - np.cos(self.qref[2])*self.qref[3] + np.sin(self.qref[2])*self.qref[4]) + c*q[5]*s*(self.kr[1]*(self.qref[1] - q[1]) + np.cos(self.qref[2])*self.qref[4] + np.sin(self.qref[2])*self.qref[3]), c**2*(np.cos(self.qref[2])*self.aref[1] - self.kr[1]*(c*q[4] - np.cos(self.qref[2])*self.qref[4] + s*q[3] - np.sin(self.qref[2])*self.qref[3]) + np.sin(self.qref[2])*self.aref[0] + np.cos(self.qref[2])*self.qref[5]*self.qref[3] - self.qref[5]*np.sin(self.qref[2])*self.qref[4]) - s*q[4]*(self.qref[5] + self.kr[2]*(Eh)) + c**2*q[5]*(self.kr[0]*(q[0] - self.qref[0]) - np.cos(self.qref[2])*self.qref[3] + np.sin(self.qref[2])*self.qref[4]) + c*s*(self.kr[0]*(c*q[3] - np.cos(self.qref[2])*self.qref[3] - s*q[4] + np.sin(self.qref[2])*self.qref[4]) - np.cos(self.qref[2])*self.aref[0] + np.sin(self.qref[2])*self.aref[1] + np.cos(self.qref[2])*self.qref[5]*self.qref[4] + self.qref[5]*np.sin(self.qref[2])*self.qref[3]) - c*q[5]*s*(self.kr[1]*(self.qref[1] - q[1]) + np.cos(self.qref[2])*self.qref[4] + np.sin(self.qref[2])*self.qref[3]),   c*(self.aref[2] - self.kr[2]*(q[5] - self.qref[5])) - q[5]*s*(self.qref[5] + self.kr[2]*(Eh)), 0, -s*q[3]*abs(q[3]), -c*q[4]*abs(q[4]), -c*q[4]*abs(q[5]), -c*q[5]*abs(q[4]), -c*q[5]*abs(q[5]), 0, 0, 0, 0],
		              [-c*q[3]*(self.kr[1]*(self.qref[1] - q[1]) + np.cos(self.qref[2])*self.qref[4] + np.sin(self.qref[2])*self.qref[3]) - s*q[3]*(self.kr[0]*(q[0] - self.qref[0]) - np.cos(self.qref[2])*self.qref[3] + np.sin(self.qref[2])*self.qref[4]),                                                                                                                                                                                                                                                                                                     s*q[4]*(self.kr[1]*(self.qref[1] - q[1]) + np.cos(self.qref[2])*self.qref[4] + np.sin(self.qref[2])*self.qref[3]) - c*q[4]*(self.kr[0]*(q[0] - self.qref[0]) - np.cos(self.qref[2])*self.qref[3] + np.sin(self.qref[2])*self.qref[4]), c*(np.cos(self.qref[2])*self.aref[1] - self.kr[1]*(c*q[4] - np.cos(self.qref[2])*self.qref[4] + s*q[3] - np.sin(self.qref[2])*self.qref[3]) + np.sin(self.qref[2])*self.aref[0] + np.cos(self.qref[2])*self.qref[5]*self.qref[3] - self.qref[5]*np.sin(self.qref[2])*self.qref[4]) + s*(self.kr[0]*(c*q[3] - np.cos(self.qref[2])*self.qref[3] - s*q[4] + np.sin(self.qref[2])*self.qref[4]) - np.cos(self.qref[2])*self.aref[0] + np.sin(self.qref[2])*self.aref[1] + np.cos(self.qref[2])*self.qref[5]*self.qref[4] + self.qref[5]*np.sin(self.qref[2])*self.qref[3]), self.aref[2] - self.kr[2]*(q[5] - self.qref[5]),                       0,                       0,                          0,              0,                 0, -q[4]*abs(q[4]), -q[4]*abs(q[5]), -q[5]*abs(q[4]), -q[5]*abs(q[5])]
		            ])

		# Control law
		u = self.kp*(R.T.dot(E)) + self.kd*Edot_body + R.T.dot(Y.dot(self.adapt))

		# Learning gradient gain
		if self.use_LS:
			# Approximate least-squares gain choice
			self.kg = self.kg - (self.kg.dot(self.ku.dot(self.Yuf.T.dot(self.Yuf))).dot(self.kg))*dt

		# Update adaptation
		adaptdot = self.kg.dot(Y.T.dot(r) + self.ku.dot(self.adapt_err))
		self.adapt = self.adapt + (adaptdot + self.last_adaptdot)*dt/2
		self.last_adaptdot = np.copy(adaptdot)

		# Update filtered prediction regressor, filtered control effort, and learning history stack
		self.update_learning(q, R, u, dt)

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

	def update_learning(self, q, R, u, dt):
		"""
		#<<< explain concurrent learning

		"""
		# Instantaneous parts of filtered prediction regressor
		c = R[0, 0]
		s = R[1, 0]
		Yuf2_now = np.array([
		                     [c*q[3], -s*q[4], -q[5]*s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		                     [s*q[3], c*q[4], c*q[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		                     [0, 0, q[4], q[5], 0, 0, 0, 0, 0, 0, 0, 0, 0]
		                   ])

		c0 = np.cos(self.q0[2])
		s0 = np.sin(self.q0[2])
		Yuf2_then = np.array([
		                      [c0*self.q0[3], -s0*self.q0[4], -self.q0[5]*s0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		                      [s0*self.q0[3], c0*self.q0[4], c0*self.q0[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		                      [0, 0, self.q0[4], self.q0[5], 0, 0, 0, 0, 0, 0, 0, 0, 0]
		                    ])

		Yuf2 = Yuf2_now - Yuf2_then

		Yuf1dot = np.array([
		                    [0, 0, 0, 0, -c*q[3]*abs(q[3]), s*q[4]*abs(q[4]), s*q[4]*abs(q[5]), q[5]*s*abs(q[4]), q[5]*s*abs(q[5]), 0, 0, 0, 0],
		                    [0, 0, 0, 0, -s*q[3]*abs(q[3]), -c*q[4]*abs(q[4]), -c*q[4]*abs(q[5]), -c*q[5]*abs(q[4]), -c*q[5]*abs(q[5]), 0, 0, 0, 0],
		                    [-q[3]*q[4], q[3]*q[4], q[5]*q[3], 0, 0, 0, 0, 0, 0, -q[4]*abs(q[4]), -q[4]*abs(q[5]), -q[5]*abs(q[4]), -q[5]*abs(q[5])]
		                  ])

		# Integral filtering of prediction regressor and control effort
		self.Yuf = self.Yuf1 + Yuf2
		# Infinite window continuous sum...
		if not self.nfilt:
			self.uf = self.uf + R.dot(u)*dt
			self.Yuf1 = self.Yuf1 + Yuf1dot*dt
		# ...or finite window push pop
		else:
			Ru = R.dot(u)
			self.uf_stack.append((Ru + self.last_Ru)*dt/2)
			self.last_Ru = np.copy(Ru)
			self.uf = (self.uf - self.uf_stack.popleft()) + self.uf_stack[-1]

			self.Yuf1_stack.append((Yuf1dot + self.last_Yuf1dot)*dt/2)
			self.last_Yuf1dot = np.copy(Yuf1dot)
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
					if np.round(eig_mins[hotseat]*100, 2) > np.round(self.history_eig*100, 2):
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

########################
	
	def make_history_pair(self, Yi, ui):
		"""
		Creates a history pair as a dictionary containing keys 'Yi' and 'ui',
		which are the filtered regressor and filtered effort for that instant.

		"""
		return {'Yi': Yi, 'ui': ui}

########################

	def get_heading_error(self, ang, R):
		"""
		Returns the heading error between a desired angle
		ang and the current orientation matrix R.

		"""
		Rdes = np.array([
		                 [np.cos(ang), -np.sin(ang), 0],
		                 [np.sin(ang),  np.cos(ang), 0],
		                 [          0,            0, 1]
		               ])
		R_err = Rdes.dot(R.T)
		return np.arctan2(R_err[1,0], R_err[0,0])

########################

	def update_ref(self, dt):
		"""
		#<<< explain trajgen

		"""
		self.path_time = self.path_time + dt

		Rref = np.array([
		                 [np.cos(self.qref[2]), -np.sin(self.qref[2]), 0],
		                 [np.sin(self.qref[2]),  np.cos(self.qref[2]), 0],
		                 [                   0,                     0, 1]
		               ])

		if self.path_type in ['waypoint', 'sequence']:
			Eref = np.concatenate((self.target[:2] - self.qref[:2], [self.get_heading_error(self.target[2], Rref)]))
			Erefdot_body = -self.qref[3:]
			uref = self.kp*(Rref.T.dot(Eref)) + self.kd*Erefdot_body
			(qrefdot, self.aref) = self.reference_dynamics(self.qref, uref, Rref)
			self.qref = self.qref + qrefdot*dt
			self.qref[2] = np.arctan2(np.sin(self.qref[2]), np.cos(self.qref[2]))
			if self.path_type == 'sequence' and self.path_time > 10:
				self.set_path(self.qref, [10, 10, 2*np.pi]*(np.random.rand(3) - 0.5), 'sequence')

		elif self.path_type == 'circle':
			r = self.target[0]  # radius, m
			T = self.target[1]  # period, s
			w = 2*np.pi/T
			c = np.cos(w*self.path_time)
			s = np.sin(w*self.path_time)
			self.qref[:2] = np.array([r*c - r, r*s], dtype=np.float32)
			self.qref[2] = w*self.path_time + np.pi/2
			self.qref[2] = np.arctan2(np.sin(self.qref[2]), np.cos(self.qref[2]))
			self.qref[3:] = np.array([w*r, 0, w], dtype=np.float32)
			self.aref = np.zeros(3)

		elif self.path_type == 'figure8':
			a = self.target[0]  # amplitude, m
			T = self.target[1]  # period, s
			w = 2*np.pi/T
			c = np.cos(w*self.path_time)
			s = np.sin(w*self.path_time)
			sq2 = np.sqrt(2)
			self.qref[:2] = [a*sq2*c/(s**2+1) - (a+1), a*sq2*c*s/(s**2+1)]
			v = [(a*sq2*w*s*(s**2 - 3))/(s**2 + 1)**2, (a*sq2*w*(3*np.cos(2*w*self.path_time) - 1))/(2*(s**2 + 1)**2)]
			self.qref[2] = np.arctan2(v[1], v[0])
			self.qref[3:] = Rref.T.dot(np.array([v[0], v[1], -(3*w*c)/(c**2 - 2)]))
			self.aref = Rref.T.dot(np.array([(a*sq2*w**2*c*(10*c**2 + c**4 - 8))/(c**2 - 2)**3,
                                             -(2*a*sq2*w**2*c*s*(3*c**2 + 2))/(s**2 + 1)**3,
			                                 (3*w**2*s*(s**2 - 3))/(s**2 + 1)**2]))
			self.aref[1] = 0

		else:
			raise ValueError("Invalid path_type.")

########################

	def reference_dynamics(self, qref, uref, Rref):
		"""
		Computes reference state derivative (qrefdot).
		Takes reference state (qref), reference control input (uref),
		and body-to-world conversion (Rref).

		"""
		# Imposed actuator saturation
		for i, mag in enumerate(abs(uref)):
			if mag > self.umaxref[i]:
				uref[i] = self.umaxref[i] * np.sign(uref[i])

		# Simple linear system to track
		aref = (uref - self.dref*qref[3:]) / self.mref
		qrefdot = np.concatenate((Rref.dot(qref[3:]) , aref))
		return (qrefdot, aref)
