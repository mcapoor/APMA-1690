function X = ising(X,T,n)

% precompute the conditional probability of a vertex being one given its neighbors
p = zeros(2,2,2,2);
for s1 = 1:2
    for s2 = 1:2
        for s3 = 1:2
            for s4 = 1:2
                a = exp((2*(s1+s2+s3+s4)-12)/T);
                b = exp(-(2*(s1+s2+s3+s4)-12)/T);
                p(s1,s2,s3,s4) = a/(a+b);
            end
        end
    end
end

% sizes
[R,C] = size(X);

% do Gibb sampling
for k = 1:n
    % do a single sweep
    for j = 1:C
        % column neighbors with wrap around
        jp1 = mod((j-1)+1,C)+1;
        jm1 = mod((j-1)-1,C)+1;
        for i = 1:R
            % row neighbors with wrap around
            ip1 = mod((i-1)+1,R)+1;
            im1 = mod((i-1)-1,R)+1;
            % get the conditional probability that X(i,j)=1 given its neighbors
            % convert {-1,+1} into {1,2} for indexing into p
            pij = p( (X(ip1,j)+3)/2, (X(im1,j)+3)/2, (X(i,jp1)+3)/2, (X(i,jm1)+3)/2 );
            % sample and convert to {-1,+1}
            X(i,j) = 2*(rand < pij)-1;
        end
    end
end
