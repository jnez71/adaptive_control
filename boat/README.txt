Below is the relationship between this code and 
http://ieeexplore.ieee.org/document/7798300/ 
as well as assorted notes for the user.

--------------------------------------------------

Trajectories that can be generated for testing:

'waypoint'
    Generates a C2 continuous trajectory that begins at the initial measured state
    and proceeds towards the pose defined by "target". The path will never achieve
    a speed (twist) larger than "vmax" or an acceleration larger than "amax".

'sequence'
    Ininitally it is the same as 'waypoint', but after 15 seconds it randomly
    selects a new target, and does so every 15 seconds forever.

'circle'
    Evident...

'figure8'
    Evident...

In general, the trajectory being tracked is considered
a reference signal, "qref".

--------------------------------------------------

Full state variable:

q  =  [eta, nu]  =  [worldframe_pose, bodyframe_twist]

--------------------------------------------------

    gains
    -----
 orig   new
 ----   ---
   k1 = kd
alpha = kr = (kp-1)/kd
   k2 = ku
gamma = kg


In the paper's control law, if you multiply out the
terms with r and e, you get

k1*edot + k1*alpha*e + e

= (k1*alpha + 1)*e + k1*edot
    proportional   derivative

= kp*e + kd*edot


If you set kg = 'LS' then it will be computed
by the least-squares method,

kgdot = -(kg)*(ku)*(Yuf')*(Yuf)*(kg)

where Yuf is scriptY.


If kf is set to anything except 0, a convolutional
filter will be used to compute scriptY instead of
the integral described by the paper. This technique
is known as torque filtering, where kf is sometimes
referred to as beta. If kf = 0, scriptY is integrated,
as described in the paper.


The integration/filter delta_t is set as the value
"window" and dictates the size of the uf (scriptu),
and Yuf (scriptY) integral/filter stacks.

--------------------------------------------------

       theta
       -----
    orig   new
    ----   ----
   m-Xud = m-wm_xu
   m-Yvd = m-wm_yv
m*xg-Yrd = m*xg-wm_yr
  Iz-Nrd = Iz-wm_nr
    Xauu = d_xuu
    Yavv = d_yvv
    Yarv = d_yrv
    Yavr = d_yvr
    Yarr = d_yrr
    Navv = d_nvv
    Narv = d_nrv
    Navr = d_nvr
    Narr = d_nrr

(note that "wm" stands for "water mass" and "d" stands for "drag")

--------------------------------------------------

   regressors
   ----------
   orig   new
   ----   ---
     Y2 = Y
scriptY = Yuf
integY4 = Yuf1
     Y3 = Yuf2
     Y4 = Yuf1dot


There are really only two regressors:

- the tracking-error regressor, I call "Y", paper calls "Y2"

- the (filtered) prediction-error regressor, I call "Yuf", paper calls "scriptY"


Yuf = Yuf1 + Yuf2 = scriptY = integ(Y4) + Y3

Yuf1dot = deriv(Yuf1) = deriv(integ(Y4)) = Y4

--------------------------------------------------
