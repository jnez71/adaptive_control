clear
clc
close all

%% measured positions and such
syms u v rPsi
nu = [u;v;rPsi];% body fixed velocity, nu

syms uDot vDot rPsiDot
nuDot = [uDot;vDot;rPsiDot]; % d/dt(nu)

syms Psi c s
R = [c, -s, 0; 
     s,  c, 0; 
     0,  0, 1];% Psi rotation

RDot = [-rPsi*s, -rPsi*c, 0; 
       rPsi*c, -rPsi*s, 0; 
            0,       0, 0];% d/dt(Psi rotation)

syms N E Psi
eta = [N;E;Psi]; % N,E,Psi positions/orientation
etaDot = simplify(R*nu);% d/dt(eta)
etaDotDot = simplify(RDot*nu+R*nuDot);% d/dt(d/dt(eta))

%% desired positions and such
syms uDes vDes rPsiDes
nuDes = [uDes;vDes;rPsiDes];% desired nu

syms uDotDes vDotDes rPsiDotDes
nuDotDes = [uDotDes;vDotDes;rPsiDotDes];% d/dt(desired nu)        
        
syms cDes sDes
RDes = [cDes, -sDes, 0; 
       sDes,  cDes, 0; 
          0,     0, 1];% desired Psi rotation

syms rPsiDes
RDotDes = [-rPsiDes*sDes, -rPsiDes*cDes, 0;
            rPsiDes*cDes, -rPsiDes*sDes, 0; 
                  0,                  0, 0];% d/dt(desired Psi rotation)

syms NDes EDes PsiDes
etaDes = [NDes;EDes;PsiDes]; % N,E,Psi positions/orientation, desired eta
etaDotDes = simplify(RDes*nuDes); % d/dt(desired eta)
etaDotDotDes = simplify(RDotDes*nuDes+RDes*nuDotDes);% d/dt(d/dt(desired eta))

%% dynamics matrices expressed in body
% inertia in body fixed frame
% M = [m-Xud,       0,       0;
%          0,   m-Yvd, mxg-Yrd;
%          0, mxg-Yrd,  Iz-Nrd]; % actual inertia
syms m11 m22 m23 m33;
M = [m11,   0,   0;
       0, m22, m23;
       0, m23, m33]; % simplified inertia and using that m32 = m23

% centipetal coriolis
% C = [                           0,            0, -(m*xg-YrDot)*rPsi-(m-YvDot)*v;
%                                 0,            0,                    (m-XuDot)*u;
%      (m*xg -YrD)*rPsi+(m-YvDot)*v, -(m-XuDot)*u,                              0]; % actual centripetal coriolis
C = [             0,      0, -m23*rPsi-m22*v;
                  0,      0,           m11*u;
     m23*rPsi+m22*v, -m11*u,               0]; % simplified centripetal coriolis using definitions from inertia matrix

% drag
% D = [-Xauu*abs(u),                           0,                           0; 
%                 0, -Yavv*abs(v)-Yarv*abs(rPsi), -Yavr*abs(v)-Yarr*abs(rPsi); 
%                 0, -Navv*abs(v)-Narv*abs(rPsi), -Navr*abs(v)-Narr*abs(rPsi)]; % actual drag
syms d11u d22v d22r d23v d23r d32v d32r d33v d33r
D = [-d11u*abs(u),                           0,                           0; 
                0, -d22v*abs(v)-d22r*abs(rPsi), -d23v*abs(v)-d23r*abs(rPsi); 
                0, -d32v*abs(v)-d32r*abs(rPsi), -d33v*abs(v)-d33r*abs(rPsi)]; % same drag but using different coefficients to make easier to read

%% dynamics matrices expressed in world
% Mstar(Psi) = R(Psi)*M*R^T(Psi)
% Mstar = simplify(R*M*transpose(R));
% Mstar = [ m11*c^2 + m22*s^2,   c*s*(m11 - m22), -m23*s]
%         [   c*s*(m11 - m22), m22*c^2 + m11*s^2,  c*m23]
%         [            -m23*s,             c*m23,    m33]
Mstar = [m11*c^2 + m22*s^2,   c*s*(m11 - m22), -m23*s;
           c*s*(m11 - m22), m22*c^2 + m11*s^2,  c*m23;
                    -m23*s,             c*m23,    m33];

% Cstar(nu,Psi) = R(Psi)*(C(nu) - M*R^T(Psi)*RDot(Psi))*R^T(Psi)
% Cstar = simplify(R * simplify(C - simplify(M*transpose(R)*RDot)) * transpose(R))
% Cstar =  [                     -c*rPsi*s*(m11 - m22)*(c^2 + s^2),                  rPsi*(m11*c^2 + m22*s^2)*(c^2 + s^2), - c*(m23*rPsi + m22*v) - m11*s*u]
%          [                 -rPsi*(m22*c^2 + m11*s^2)*(c^2 + s^2),                      c*rPsi*s*(m11 - m22)*(c^2 + s^2),   c*m11*u - s*(m23*rPsi + m22*v)]
%          [ c*(m23*rPsi + m22*v - m23*rPsi*(c^2 + s^2)) + m11*s*u, s*(m23*rPsi + m22*v - m23*rPsi*(c^2 + s^2)) - c*m11*u,                                0]
% remove the (c^2+s^2)
% Cstar =  [                     -c*rPsi*s*(m11 - m22),                  rPsi*(m11*c^2 + m22*s^2), - c*(m23*rPsi + m22*v) - m11*s*u]
%          [                 -rPsi*(m22*c^2 + m11*s^2),                      c*rPsi*s*(m11 - m22),   c*m11*u - s*(m23*rPsi + m22*v)]
%          [ c*(m23*rPsi + m22*v - m23*rPsi) + m11*s*u, s*(m23*rPsi + m22*v - m23*rPsi) - c*m11*u,                                0]
% cancel out remaining terms
% Cstar =  [     -c*rPsi*s*(m11 - m22), rPsi*(m11*c^2 + m22*s^2), -c*(m23*rPsi + m22*v) - m11*s*u]
%          [ -rPsi*(m22*c^2 + m11*s^2),     c*rPsi*s*(m11 - m22),  c*m11*u - s*(m23*rPsi + m22*v)]
%          [         c*m22*v + m11*s*u,        s*m22*v - c*m11*u,                               0]
Cstar =  [    -c*rPsi*s*(m11 - m22), rPsi*(m11*c^2 + m22*s^2), -c*(m23*rPsi + m22*v) - m11*s*u;
          -rPsi*(m22*c^2 + m11*s^2),     c*rPsi*s*(m11 - m22),  c*m11*u - s*(m23*rPsi + m22*v);
                  c*m22*v + m11*s*u,        s*m22*v - c*m11*u,                               0];

% Dstar(nu,Psi) = R(Psi)*D(nu)*R^T(Psi)
% Dstar = simplify(R*D*transpose(R));
% Dstar = [ - d11u*abs(u)*c^2 + (- d22r*abs(rPsi) - d22v*abs(v))*s^2,       c*s*(d22r*abs(rPsi) - d11u*abs(u) + d22v*abs(v)),  s*(d23r*abs(rPsi) + d23v*abs(v))]
%         [         c*s*(d22r*abs(rPsi) - d11u*abs(u) + d22v*abs(v)), (- d22r*abs(rPsi) - d22v*abs(v))*c^2 - d11u*abs(u)*s^2, -c*(d23r*abs(rPsi) + d23v*abs(v))]
%         [                         s*(d32r*abs(rPsi) + d32v*abs(v)),                      -c*(d32r*abs(rPsi) + d32v*abs(v)),    - d33r*abs(rPsi) - d33v*abs(v)]
Dstar = [-d11u*abs(u)*c^2 + (- d22r*abs(rPsi) - d22v*abs(v))*s^2,      c*s*(d22r*abs(rPsi) - d11u*abs(u) + d22v*abs(v)),  s*(d23r*abs(rPsi) + d23v*abs(v));
                c*s*(d22r*abs(rPsi) - d11u*abs(u) + d22v*abs(v)), (-d22r*abs(rPsi) - d22v*abs(v))*c^2 - d11u*abs(u)*s^2, -c*(d23r*abs(rPsi) + d23v*abs(v));
                                s*(d32r*abs(rPsi) + d32v*abs(v)),                     -c*(d32r*abs(rPsi) + d32v*abs(v)),    - d33r*abs(rPsi) - d33v*abs(v)];

%% unknowns
theta = transpose([m11, m22, m23, m33, d11u, d22v, d22r, d23v, d23r, d32v, d32r, d33v, d33r]);

%% Y1(nu,Psi,etaDot,etaDotDot)*theta = Mstar(Psi)*etaDotDot + Cstar(nu,Psi)*etaDot + Dstar(nu,Psi)*etaDot
Y1theta = collect( simplify( simplify(Mstar*etaDotDot) + simplify(Cstar*etaDot) + simplify(Dstar*etaDot) ) ,theta);
% (uDot*c^3 + uDot*c*s^2 - rPsi*u*s)*m11 + (-vDot*c^2*s - rPsi*v*c - vDot*s^3)*m22 +                         (-c*rPsi^2 - rPsiDot*s)*m23 +       0*m33 + (-u*abs(u)*c^3 - u*abs(u)*c*s^2)*d11u +  (v*abs(v)*c^2*s + v*abs(v)*s^3)*d22v +  (v*abs(rPsi)*c^2*s + v*abs(rPsi)*s^3)*d22r +  (rPsi*s*abs(v))*d23v +  (rPsi*s*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
% (uDot*c^2*s + rPsi*u*c + uDot*s^3)*m11 +  (vDot*c^3 + vDot*c*s^2 - rPsi*v*s)*m22 +                         (-s*rPsi^2 + c*rPsiDot)*m23 +       0*m33 + (-u*abs(u)*c^2*s - u*abs(u)*s^3)*d11u + (-v*abs(v)*c^3 - v*abs(v)*c*s^2)*d22v + (-v*abs(rPsi)*c^3 - v*abs(rPsi)*c*s^2)*d22r + (-c*rPsi*abs(v))*d23v + (-c*rPsi*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
%                 (-u*v*(c^2 + s^2))*m11 +                   (u*v*(c^2 + s^2))*m22 + (c^2*vDot + s^2*vDot + c^2*rPsi*u + rPsi*s^2*u)*m23 + rPsiDot*m33 +                                0*d11u +                                0*d22v +                                      0*d22r +                0*d23v +                   0*d23r + (s*abs(v)*(c*u - s*v) - c*abs(v)*(c*v + s*u))*d32v + (s*abs(rPsi)*(c*u - s*v) - c*abs(rPsi)*(c*v + s*u))*d32r + (-rPsi*abs(v))*d33v + (-rPsi*abs(rPsi))*d33r
% there are some (c^2 + s^2)
% (uDot*c^3 + uDot*c*s^2 - rPsi*u*s)*m11 + (-vDot*c^2*s - rPsi*v*c - vDot*s^3)*m22 +                         (-c*rPsi^2 - rPsiDot*s)*m23 +       0*m33 + (-u*abs(u)*c^3 - u*abs(u)*c*s^2)*d11u +  (v*abs(v)*c^2*s + v*abs(v)*s^3)*d22v +  (v*abs(rPsi)*c^2*s + v*abs(rPsi)*s^3)*d22r +  (rPsi*s*abs(v))*d23v +  (rPsi*s*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
% (uDot*c^2*s + rPsi*u*c + uDot*s^3)*m11 +  (vDot*c^3 + vDot*c*s^2 - rPsi*v*s)*m22 +                         (-s*rPsi^2 + c*rPsiDot)*m23 +       0*m33 + (-u*abs(u)*c^2*s - u*abs(u)*s^3)*d11u + (-v*abs(v)*c^3 - v*abs(v)*c*s^2)*d22v + (-v*abs(rPsi)*c^3 - v*abs(rPsi)*c*s^2)*d22r + (-c*rPsi*abs(v))*d23v + (-c*rPsi*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
%                               -u*v*m11 +                                 u*v*m22 + (c^2*vDot + s^2*vDot + c^2*rPsi*u + rPsi*s^2*u)*m23 + rPsiDot*m33 +                                0*d11u +                                0*d22v +                                      0*d22r +                0*d23v +                   0*d23r + (s*abs(v)*(c*u - s*v) - c*abs(v)*(c*v + s*u))*d32v + (s*abs(rPsi)*(c*u - s*v) - c*abs(rPsi)*(c*v + s*u))*d32r + (-rPsi*abs(v))*d33v + (-rPsi*abs(rPsi))*d33r
% can factor out some other stuff to get more c^2+s^2
% (uDot*c*(c^2 + s^2) - rPsi*u*s)*m11 + (-vDot*s*(c^2 + s^2) - rPsi*v*c)*m22 +                 (-c*rPsi^2 - rPsiDot*s)*m23 +       0*m33 + (-u*abs(u)*c*(c^2 + s^2))*d11u +   (v*abs(v)*s*(c^2 + s^2))*d22v +  (v*abs(rPsi)*s*(c^2 + s^2))*d22r +  (rPsi*s*abs(v))*d23v +  (rPsi*s*abs(rPsi))*d23r +                                                        0*d32v +                                                                 0*d32r +              0*d33v +                 0*d33r
% (uDot*s*(c^2 + s^2) + rPsi*u*c)*m11 +  (vDot*c*(c^2 + s^2) - rPsi*v*s)*m22 +                 (-s*rPsi^2 + c*rPsiDot)*m23 +       0*m33 + (-u*abs(u)*s*(c^2 + s^2))*d11u +  (-v*abs(v)*c*(c^2 + s^2))*d22v + (-v*abs(rPsi)*c*(c^2 + s^2))*d22r + (-c*rPsi*abs(v))*d23v + (-c*rPsi*abs(rPsi))*d23r +                                                        0*d32v +                                                                 0*d32r +              0*d33v +                 0*d33r
%                            -u*v*m11 +                              u*v*m22 + (vDot*(c^2 + s^2) + rPsi*u*(c^2 + s^2))*m23 + rPsiDot*m33 +                         0*d11u +                         0*d22v +                             0*d22r +                0*d23v +                   0*d23r + ( -abs(v)*v*(s^2 + c^2) + (s*abs(v)*c*u - c*abs(v)*s*u))*d32v + ( -abs(rPsi)*v*(s^2 + c^2) + (s*abs(rPsi)*c*u - c*abs(rPsi)*s*u))*d32r + (-rPsi*abs(v))*d33v + (-rPsi*abs(rPsi))*d33r
% remove the c^2+s^2 and cancle out stuff
% (uDot*c - rPsi*u*s)*m11 + (-vDot*s - rPsi*v*c)*m22 + (-c*rPsi^2 - rPsiDot*s)*m23 +       0*m33 + -u*abs(u)*c*d11u +   v*abs(v)*s*d22v +  v*abs(rPsi)*s*d22r +  rPsi*s*abs(v)*d23v +  rPsi*s*abs(rPsi)*d23r +         0*d32v +            0*d32r +            0*d33v +               0*d33r
% (uDot*s + rPsi*u*c)*m11 +  (vDot*c - rPsi*v*s)*m22 + (-s*rPsi^2 + c*rPsiDot)*m23 +       0*m33 + -u*abs(u)*s*d11u +  -v*abs(v)*c*d22v + -v*abs(rPsi)*c*d22r + -c*rPsi*abs(v)*d23v + -c*rPsi*abs(rPsi)*d23r +         0*d32v +            0*d32r +            0*d33v +               0*d33r
%                -u*v*m11 +                  u*v*m22 +         (vDot + rPsi*u)*m23 + rPsiDot*m33 +           0*d11u +            0*d22v +              0*d22r +              0*d23v +                 0*d23r + -abs(v)*v*d32v + -abs(rPsi)*v*d32r + -rPsi*abs(v)*d33v + -rPsi*abs(rPsi)*d33r
% Y1 = [(uDot*c - rPsi*u*s), (-vDot*s - rPsi*v*c),  (-c*rPsi^2 - rPsiDot*s),       0, -u*abs(u)*c,   v*abs(v)*s,  v*abs(rPsi)*s,  rPsi*s*abs(v),  rPsi*s*abs(rPsi),         0,            0,            0,               0;
%       (uDot*s + rPsi*u*c),  (vDot*c - rPsi*v*s),  (-s*rPsi^2 + c*rPsiDot),       0, -u*abs(u)*s,  -v*abs(v)*c, -v*abs(rPsi)*c, -c*rPsi*abs(v), -c*rPsi*abs(rPsi),         0,            0,            0,               0;
%                      -u*v,                  u*v,          (vDot + rPsi*u), rPsiDot,           0,            0,              0,              0,                 0, -abs(v)*v, -abs(rPsi)*v, -rPsi*abs(v), -rPsi*abs(rPsi)]

%% Y2(nu,Psi,eta,etaDot,etaDes,etaDotDes,etaDotDotDes)*theta = Mstar*(etaDotDotDes + alpha*eDot) + Cstar*(etaDotDes + alpha*e) + Dstar*etaDot
syms alpha1 alpha2 alpha3
alpha = diag([alpha1 alpha2 alpha3]); % alpha
e = simplify(etaDes-eta);% error
eDot = simplify(etaDotDes-etaDot);% d/dt(error)
Y2theta = collect( simplify( simplify(Mstar * simplify(etaDotDotDes+alpha*eDot) ) + simplify(Cstar * simplify(etaDotDes + alpha*e) ) + simplify(Dstar*etaDot) ), theta);
% (c^2*rPsi*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - s*u*(rPsiDes - alpha3*(Psi - PsiDes)) - c^2*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) + c*s*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + c*rPsi*s*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes))*m11 + (rPsi*s^2*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - c*v*(rPsiDes - alpha3*(Psi - PsiDes)) - s^2*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) - c*s*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) - c*rPsi*s*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes))*m22 +                                                                                                                                                       (- s*(rPsiDotDes - alpha3*(rPsi - rPsiDes)) - c*rPsi*(rPsiDes - alpha3*(Psi - PsiDes)))*m23 +                                      0*m33 + (- c^2*abs(u)*(c*u - s*v) - c*s*abs(u)*(c*v + s*u))*d11u + (c*s*abs(v)*(c*v + s*u) - s^2*abs(v)*(c*u - s*v))*d22v + (c*s*abs(rPsi)*(c*v + s*u) - s^2*abs(rPsi)*(c*u - s*v))*d22r +  (rPsi*s*abs(v))*d23v +  (rPsi*s*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
% (s^2*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + c*u*(rPsiDes - alpha3*(Psi - PsiDes)) - c*s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) + rPsi*s^2*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes) + c*rPsi*s*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes))*m11 + (c^2*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) - s*v*(rPsiDes - alpha3*(Psi - PsiDes)) + c^2*rPsi*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes) + c*s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) - c*rPsi*s*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes))*m22 +                                                                                                                                                         (c*(rPsiDotDes - alpha3*(rPsi - rPsiDes)) - rPsi*s*(rPsiDes - alpha3*(Psi - PsiDes)))*m23 +                                      0*m33 + (- s^2*abs(u)*(c*v + s*u) - c*s*abs(u)*(c*u - s*v))*d11u + (c*s*abs(v)*(c*u - s*v) - c^2*abs(v)*(c*v + s*u))*d22v + (c*s*abs(rPsi)*(c*u - s*v) - c^2*abs(rPsi)*(c*v + s*u))*d22r + (-c*rPsi*abs(v))*d23v + (-c*rPsi*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
%                                                                                                                                                                                                                                                                                                   (- c*u*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - s*u*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes))*m11 +                                                                                                                                                                                                                                                                                                     (s*v*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - c*v*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes))*m22 + (c*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes))*m23 + (rPsiDotDes - alpha3*(rPsi - rPsiDes))*m33 +                                                   0*d11u +                                                 0*d22v +                                                       0*d22r +                0*d23v +                   0*d23r + (s*abs(v)*(c*u - s*v) - c*abs(v)*(c*v + s*u))*d32v + (s*abs(rPsi)*(c*u - s*v) - c*abs(rPsi)*(c*v + s*u))*d32r + (-rPsi*abs(v))*d33v + (-rPsi*abs(rPsi))*d33r
% didnt really see anything that could cancel
% (c^2*rPsi*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - s*u*(rPsiDes - alpha3*(Psi - PsiDes)) - c^2*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) + c*s*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + c*rPsi*s*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes))*m11 + (rPsi*s^2*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - c*v*(rPsiDes - alpha3*(Psi - PsiDes)) - s^2*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) - c*s*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) - c*rPsi*s*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes))*m22 +                                                                                                                                                       (- s*(rPsiDotDes - alpha3*(rPsi - rPsiDes)) - c*rPsi*(rPsiDes - alpha3*(Psi - PsiDes)))*m23 +                                      0*m33 + (- c^2*abs(u)*(c*u - s*v) - c*s*abs(u)*(c*v + s*u))*d11u + (c*s*abs(v)*(c*v + s*u) - s^2*abs(v)*(c*u - s*v))*d22v + (c*s*abs(rPsi)*(c*v + s*u) - s^2*abs(rPsi)*(c*u - s*v))*d22r +  (rPsi*s*abs(v))*d23v +  (rPsi*s*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
% (s^2*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + c*u*(rPsiDes - alpha3*(Psi - PsiDes)) - c*s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) + rPsi*s^2*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes) + c*rPsi*s*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes))*m11 + (c^2*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) - s*v*(rPsiDes - alpha3*(Psi - PsiDes)) + c^2*rPsi*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes) + c*s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) - c*rPsi*s*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes))*m22 +                                                                                                                                                         (c*(rPsiDotDes - alpha3*(rPsi - rPsiDes)) - rPsi*s*(rPsiDes - alpha3*(Psi - PsiDes)))*m23 +                                      0*m33 + (- s^2*abs(u)*(c*v + s*u) - c*s*abs(u)*(c*u - s*v))*d11u + (c*s*abs(v)*(c*u - s*v) - c^2*abs(v)*(c*v + s*u))*d22v + (c*s*abs(rPsi)*(c*u - s*v) - c^2*abs(rPsi)*(c*v + s*u))*d22r + (-c*rPsi*abs(v))*d23v + (-c*rPsi*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
%                                                                                                                                                                                                                                                                                                   (- c*u*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - s*u*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes))*m11 +                                                                                                                                                                                                                                                                                                     (s*v*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - c*v*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes))*m22 + (c*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes))*m23 + (rPsiDotDes - alpha3*(rPsi - rPsiDes))*m33 +                                                   0*d11u +                                                 0*d22v +                                                       0*d22r +                0*d23v +                   0*d23r + (s*abs(v)*(c*u - s*v) - c*abs(v)*(c*v + s*u))*d32v + (s*abs(rPsi)*(c*u - s*v) - c*abs(rPsi)*(c*v + s*u))*d32r + (-rPsi*abs(v))*d33v + (-rPsi*abs(rPsi))*d33r
% Y2 = [(c^2*rPsi*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - s*u*(rPsiDes - alpha3*(Psi - PsiDes)) - c^2*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) + c*s*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + c*rPsi*s*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes)), (rPsi*s^2*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - c*v*(rPsiDes - alpha3*(Psi - PsiDes)) - s^2*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) - c*s*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) - c*rPsi*s*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes)),                                                                                                                                                       (- s*(rPsiDotDes - alpha3*(rPsi - rPsiDes)) - c*rPsi*(rPsiDes - alpha3*(Psi - PsiDes))),                                      0, (- c^2*abs(u)*(c*u - s*v) - c*s*abs(u)*(c*v + s*u)), (c*s*abs(v)*(c*v + s*u) - s^2*abs(v)*(c*u - s*v)), (c*s*abs(rPsi)*(c*v + s*u) - s^2*abs(rPsi)*(c*u - s*v)),  (rPsi*s*abs(v)),  (rPsi*s*abs(rPsi)),                                             0,                                                   0,              0,                 0;
%       (s^2*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + c*u*(rPsiDes - alpha3*(Psi - PsiDes)) - c*s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) + rPsi*s^2*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes) + c*rPsi*s*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes)), (c^2*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) - s*v*(rPsiDes - alpha3*(Psi - PsiDes)) + c^2*rPsi*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes) + c*s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes) - c*rPsi*s*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes)),                                                                                                                                                         (c*(rPsiDotDes - alpha3*(rPsi - rPsiDes)) - rPsi*s*(rPsiDes - alpha3*(Psi - PsiDes))),                                      0, (- s^2*abs(u)*(c*v + s*u) - c*s*abs(u)*(c*u - s*v)), (c*s*abs(v)*(c*u - s*v) - c^2*abs(v)*(c*v + s*u)), (c*s*abs(rPsi)*(c*u - s*v) - c^2*abs(rPsi)*(c*v + s*u)), (-c*rPsi*abs(v)), (-c*rPsi*abs(rPsi)),                                             0,                                                   0,              0,                 0;
%                                                                                                                                                                                                                                                                                                         (- c*u*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - s*u*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes)),                                                                                                                                                                                                                                                                                                     (s*v*(alpha2*(EDes - E) + cDes*vDes + sDes*uDes) - c*v*(alpha1*(N - NDes) - cDes*uDes + sDes*vDes)), (c*(cDes*vDotDes - alpha2*(c*v - cDes*vDes + s*u - sDes*uDes) + sDes*uDotDes + cDes*rPsiDes*uDes - rPsiDes*sDes*vDes) + s*(alpha1*(c*u - cDes*uDes - s*v + sDes*vDes) - cDes*uDotDes + sDes*vDotDes + cDes*rPsiDes*vDes + rPsiDes*sDes*uDes)), (rPsiDotDes - alpha3*(rPsi - rPsiDes)),                                                   0,                                                 0,                                                       0,                0,                   0, (s*abs(v)*(c*u - s*v) - c*abs(v)*(c*v + s*u)), (s*abs(rPsi)*(c*u - s*v) - c*abs(rPsi)*(c*v + s*u)), (-rPsi*abs(v)), (-rPsi*abs(rPsi))];
Y2 = simplify(extract_Y(Y2theta, theta))

%% Y3(Psi,etaDot)*theta = Mstar(Psi(t))*etaDot(t) - Mstar(Psi(t-delT))*etaDot(t-delT) 
Y3thetaTerm1 = collect(simplify(Mstar*etaDot),theta);
% (c^2*(c*u - s*v) + c*s*(c*v + s*u))*m11 + (s^2*(c*u - s*v) - c*s*(c*v + s*u))*m22 +        (-rPsi*s)*m23 +    0*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
% (s^2*(c*v + s*u) + c*s*(c*u - s*v))*m11 + (c^2*(c*v + s*u) - c*s*(c*u - s*v))*m22 +         (c*rPsi)*m23 +    0*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
%                                   0*m11 +                                   0*m22 +  (v*c^2 + v*s^2)*m23 + rPsi*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
% can simplify a bit
% (c*u*(c^2 + s^2) + (-c^2*s*v + c^2*s*v))*m11 + (s^2*(c*u - s*v) - c*s*(c*v + s*u))*m22 +        (-rPsi*s)*m23 +    0*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
% (s*u*(c^2* + s^2) + (s^2*c*v - c*s^2*v))*m11 + (c^2*(c*v + s*u) - c*s*(c*u - s*v))*m22 +         (c*rPsi)*m23 +    0*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
%                                        0*m11 +                                   0*m22 +  (v*(c^2 + s^2))*m23 + rPsi*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
% remove the terms
% (c*u)*m11 + (s^2*(c*u - s*v) - c*s*(c*v + s*u))*m22 +        (-rPsi*s)*m23 +    0*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
% (s*u)*m11 + (c^2*(c*v + s*u) - c*s*(c*u - s*v))*m22 +         (c*rPsi)*m23 +    0*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
%     0*m11 +                                   0*m22 +              (v)*m23 + rPsi*m33 + 0*d11u + 0*d22v + 0*d22r + 0*d23v + 0*d23r + 0*d32v + 0*d32r + 0*d33v + 0*d33r
% Y3Term1 = [(c*u), (s^2*(c*u - s*v) - c*s*(c*v + s*u)),        (-rPsi*s),    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
%            (s*u), (c^2*(c*v + s*u) - c*s*(c*u - s*v)),         (c*rPsi),    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
%                0,                                   0,                v, rPsi, 0, 0, 0, 0, 0, 0, 0, 0, 0];

Y3Term1 = simplify(extract_Y(Y3thetaTerm1, theta))

%% Y4(nu,Psi,etaDot)*theta = -MstarDot*etaDot+Cstar*etaDot+Dstar*etaDot
MstarDot = simplify( simplify(RDot*M*transpose(R)) + simplify(R*M*transpose(RDot)) );
Y4theta = collect( simplify( simplify(-MstarDot*etaDot) + simplify(Cstar*etaDot) + simplify(Dstar*etaDot)) ,theta);
%    (rPsi*u*c^2*s + rPsi*u*s^3 - rPsi*u*s)*m11 + (rPsi*v*c^3 + rPsi*v*c*s^2 - rPsi*v*c)*m22 +                    0*m23 + 0*m33 + (- u*abs(u)*c^3 - u*abs(u)*c*s^2)*d11u +  (v*abs(v)*c^2*s + v*abs(v)*s^3)*d22v +  (v*abs(rPsi)*c^2*s + v*abs(rPsi)*s^3)*d22r +  (rPsi*s*abs(v))*d23v +  (rPsi*s*abs(rPsi))*d23r                                             + 0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
%  (- rPsi*u*c^3 - rPsi*u*c*s^2 + rPsi*u*c)*m11 + (rPsi*v*c^2*s + rPsi*v*s^3 - rPsi*v*s)*m22 +                    0*m23 + 0*m33 + (- u*abs(u)*c^2*s - u*abs(u)*s^3)*d11u + (-v*abs(v)*c^3 - v*abs(v)*c*s^2)*d22v + (-v*abs(rPsi)*c^3 - v*abs(rPsi)*c*s^2)*d22r + (-c*rPsi*abs(v))*d23v + (-c*rPsi*abs(rPsi))*d23r                                             + 0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
%                        (-u*v*(c^2 + s^2))*m11 +                      (u*v*(c^2 + s^2))*m22 + (rPsi*u*(c^2 + s^2))*m23 + 0*m33 +                                 0*d11u +                                0*d22v +                                      0*d22r +                0*d23v +                   0*d23r + (s*abs(v)*(c*u - s*v) - c*abs(v)*(c*v + s*u))*d32v + (s*abs(rPsi)*(c*u - s*v) - c*abs(rPsi)*(c*v + s*u))*d32r + (-rPsi*abs(v))*d33v + (-rPsi*abs(rPsi))*d33r
% simplify some
%    (rPsi*u*c^2*s + rPsi*u*s^3 - rPsi*u*s)*m11 + (rPsi*v*c^3 + rPsi*v*c*s^2 - rPsi*v*c)*m22 +                    0*m23 + 0*m33 + (- u*abs(u)*c^3 - u*abs(u)*c*s^2)*d11u +  (v*abs(v)*c^2*s + v*abs(v)*s^3)*d22v +  (v*abs(rPsi)*c^2*s + v*abs(rPsi)*s^3)*d22r +  (rPsi*s*abs(v))*d23v +  (rPsi*s*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
%  (- rPsi*u*c^3 - rPsi*u*c*s^2 + rPsi*u*c)*m11 + (rPsi*v*c^2*s + rPsi*v*s^3 - rPsi*v*s)*m22 +                    0*m23 + 0*m33 + (- u*abs(u)*c^2*s - u*abs(u)*s^3)*d11u + (-v*abs(v)*c^3 - v*abs(v)*c*s^2)*d22v + (-v*abs(rPsi)*c^3 - v*abs(rPsi)*c*s^2)*d22r + (-c*rPsi*abs(v))*d23v + (-c*rPsi*abs(rPsi))*d23r +                                             0*d32v +                                                   0*d32r +              0*d33v +                 0*d33r
%                                    (-u*v)*m11 +                                  (u*v)*m22 +             (rPsi*u)*m23 + 0*m33 +                                 0*d11u +                                0*d22v +                                      0*d22r +                0*d23v +                   0*d23r + (s*abs(v)*(c*u - s*v) - c*abs(v)*(c*v + s*u))*d32v + (s*abs(rPsi)*(c*u - s*v) - c*abs(rPsi)*(c*v + s*u))*d32r + (-rPsi*abs(v))*d33v + (-rPsi*abs(rPsi))*d33r
% Y4 = [(rPsi*u*c^2*s + rPsi*u*s^3 - rPsi*u*s), (rPsi*v*c^3 + rPsi*v*c*s^2 - rPsi*v*c),                    0, 0, (- u*abs(u)*c^3 - u*abs(u)*c*s^2),  (v*abs(v)*c^2*s + v*abs(v)*s^3),  (v*abs(rPsi)*c^2*s + v*abs(rPsi)*s^3),  (rPsi*s*abs(v)),  (rPsi*s*abs(rPsi)),                                             0,                                                   0,              0,                 0;
%      (-rPsi*u*c^3 - rPsi*u*c*s^2 + rPsi*u*c), (rPsi*v*c^2*s + rPsi*v*s^3 - rPsi*v*s),                    0, 0, (- u*abs(u)*c^2*s - u*abs(u)*s^3), (-v*abs(v)*c^3 - v*abs(v)*c*s^2), (-v*abs(rPsi)*c^3 - v*abs(rPsi)*c*s^2), (-c*rPsi*abs(v)), (-c*rPsi*abs(rPsi)),                                             0,                                                   0,              0,                 0;
%                                       (-u*v),                                  (u*v),             (rPsi*u), 0,                                 0,                                0,                                      0,                0,                   0, (s*abs(v)*(c*u - s*v) - c*abs(v)*(c*v + s*u)), (s*abs(rPsi)*(c*u - s*v) - c*abs(rPsi)*(c*v + s*u)), (-rPsi*abs(v)), (-rPsi*abs(rPsi))];
                                      
Y4 = simplify(extract_Y(Y4theta, theta))                                     
                                      
                                      
                                      
                                      
                                   
