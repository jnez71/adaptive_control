clear all
clc

syms m1 m2 L1 L2 g q1 q2 w1 w2 a1 a2;

M = [(m1+m2)*L1^2 + m2*L2^2 + 2*m2*L1*L2*cos(q2), m2*L2^2 + m2*L1*L2*cos(q2); m2*L2^2 + m2*L1*L2*cos(q2), m2*L2^2];

V = [-m2*L1*L2*(2*w1*w2+w2^2)*sin(q2); m2*L1*L2*w1^2*sin(q2)];

G = [g*(m1+m2)*L1*cos(q1) + m2*g*L2*cos(q1+q2); m2*g*L2*cos(q1+q2)];

EQ = M*[a1; a2] + V + G;

EQ_ex = expand(EQ);

% FOR FULL DYNAMICS Y*THETA = u

theta = [L1*g*(m1+m2); L1^2*(m1+m2); L2*g*m2; L1*L2*m2; L2^2*m2]

Y_full1 = [cos(q1), a1, cos(q1+q2), cos(q2)*(2*a1+a2)-sin(q2)*(w2^2+2*w1*w2), a1+a2];

Y_full2 = [0, 0, cos(q1+q2), a1*cos(q2)+w1^2*sin(q2), a1+a2];

Y_full = [Y_full1; Y_full2];

check_full = simplify(expand(Y_full*theta)==EQ_ex)

disp('')
disp('Y_full*THETA = M*qddot + V + G = u')
disp('where:')
disp('')
Y_full

% FOR ERROR DYNAMICS M*rdot = Y*THETA - u

syms w1d w2d a1d a2d edot1 edot2 th1 th2 th3 th4 th5 p1 p2;

Y_removed = simplify(subs(subs(Y_full, a1, 0), a2, 0));

check_removed = simplify(M*[a1; a2] + Y_removed*theta == EQ_ex);

M_simp = [th2 + th5 + 2*th4*cos(q2), th5 + th4*cos(q2); th5 + th4*cos(q2), th5];

tracking_terms = M_simp*[a1d; a2d] + M_simp*[p1*edot1; p2*edot2];

Ycol1 = simplify(subs(tracking_terms, [th2, th3, th4, th5], zeros(1, 4)) / th1);
Ycol2 = simplify(subs(tracking_terms, [th3, th4, th5, th1], zeros(1, 4)) / th2);
Ycol3 = simplify(subs(tracking_terms, [th4, th5, th1, th2], zeros(1, 4)) / th3);
Ycol4 = simplify(subs(tracking_terms, [th5, th1, th2, th3], zeros(1, 4)) / th4);
Ycol5 = simplify(subs(tracking_terms, [th1, th2, th3, th4], zeros(1, 4)) / th5);

Y_tracking = [Ycol1, Ycol2, Ycol3, Ycol4, Ycol5];

check_tracking = simplify(expand(Y_tracking*theta) == expand(M*[a1d; a2d] + M*[p1*edot1; p2*edot2]));

Y_err = subs(Y_tracking+Y_removed, [edot1, edot2], [w1d-w1, w2d-w2]);

check_err = simplify(expand(Y_err*theta) == expand(M*[a1d; a2d] + M*diag([p1,p2])*[w1d-w1; w2d-w2] + V + G))

disp('')
disp('M*rdot = M*qddot_d + Medot + V + G - u = Y*THETA - u')
disp('where:')
disp('')
Y = Y_err

% FOR TORQUE FILTERING
syms b

Mdot = [-2*m2*L1*L2*sin(q2)*w2, -m2*L1*L2*sin(q2)*w2; -m2*L1*L2*sin(q2)*w2, 0];

Wu1 = expand(simplify(V + G - (b*M + Mdot)*[w1;w2]));

expand(Mdot*[w1;w2]);
Y_Mdotw = sym(zeros(2,5));
Y_Mdotw(1,4) = -sin(q2)*w2^2-2*w1*w2*sin(q2);
Y_Mdotw(2,4) = -w1*w2*sin(q2);

expand(b*M*[w1;w2]);
Y_bMw = sym(zeros(2,5));
Y_bMw(1,2) = b*w1;
Y_bMw(1,4) = (2*w1+w2)*b*cos(q2);
Y_bMw(1,5) = b*(w1+w2);
Y_bMw(2,4) = b*w1*cos(q2);
Y_bMw(2,5) = b*(w1+w2);

Y_Ma = simplify(subs(Y_bMw, [w1,w2], [a1,a2])/b);

Y_uf1 = simplify(Y_full - Y_Ma - Y_bMw - Y_Mdotw);

check_Yuf1 = simplify(expand(Y_uf1*theta) == Wu1)

disp('')
disp('Yuf1dot*theta + b*Yuf1 = V + G - (b*M + Mdot)*qdot')
disp('where:')
disp('')
Y_uf1

syms ex

disp('also:')
Y_uf2 = simplify(Y_bMw/b)






