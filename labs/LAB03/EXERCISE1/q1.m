for m = [4,5,8]
A = zeros(21,(m+1));
r = 1;
c = 1;
for x = -5:0.5:5
    for i = 0:m
    A(r,c) = x^i;
    c = c+1;
    end
c = 1;
r = r+1;
end

i = -10;
b = zeros (21,1);
for j = 1:21
b(j,1) = 1 / (1 + (i/2)^2);
i = i+1;
end

res = lsqr(A,b);

p = res';

syms x k
G(x) = sum(p.*subs(x^k,k,0:m));

y_1 = @(x) G(x)
x_1 = -5:0.5:5;
plot(x_1,y_1(x_1))
hold on
end

x_1 = -5:0.5:5;
y_2 = q1_func(x_1);

plot(x_1,y_2)
