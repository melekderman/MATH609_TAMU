function y_2 = q1_func(x)
    y_2 = zeros(length(x),1)';
    y_2 =  1 ./ (1 + (x).^2);
end