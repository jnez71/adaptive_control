function [ Y ] = extract_Y( YT, T )
% Extract the regressor (Y) from a collected
% Y*Theta (YT) for a given Theta (T).

Y = sym(zeros(size(YT, 1), length(T)));

for i = 1:length(T)
    
    Tnot = reshape(T, [1, length(T)]);
    Tnot(i) = [];
    Z = zeros(1, length(Tnot));
    Y(:, i) = simplify(subs(YT, Tnot, Z) / T(i));
    
end

end
