function [Dist,D,k,w]=dtw(t,r,max_dist)
% fprintf('DTW:\n');
% fprintf('t: '); fprintf('%.1f ', t); fprintf('\n');
% fprintf('r: '); fprintf('%.1f ', r); fprintf('\n');

%Dynamic Time Warping Algorithm
%Dist is unnormalized distance between t and r
%D is the accumulated distance matrix
%k is the normalizing factor
%w is the optimal path
%t is the vector you are testing against
%r is the vector you are testing
[rows,N]=size(t); 

[~,M]=size(r);
d = zeros(N,M);  
for n=1:N
    for m=1:M
        d(n,m)=sqrt(sum((t(:,n)-r(:,m)).^2));
    end
end
%d = distance(t, r)/rows;
%disp(d);
% d=(repmat(t(:),1,M)-repmat(r(:)',N,1)).^2; %this replaces the nested for loops from above Thanks Georg Schmitz 

D=inf(size(d));
D(1,1)=d(1,1);

for n=2:N
    D(n,1)=d(n,1)+D(n-1,1);
end
for m=2:M
    D(1,m)=d(1,m)+D(1,m-1);
end
encode_cost = 1;
mcost = encode_cost * std(t(:)) * log2(M);
ncost = encode_cost * std(r(:)) * log2(N);
for n=2:N
    m_min = max(2, n-max_dist);
    m_max = min(M, n+max_dist);
    for m=m_min:m_max
        D(n,m)=d(n,m)+min(min(D(n-1,m)+mcost, D(n-1,m-1)), D(n,m-1)+ncost);
       
    end
end

Dist=D(N,M);
n=N;
m=M;
k=1;
w=[];
w(1,:)=[N,M];
while ((n+m)~=2)
    if (n-1)==0
        m=m-1;
    elseif (m-1)==0
        n=n-1;
    else 
      [values,number]=min([D(n-1,m),D(n,m-1),D(n-1,m-1)]);
      switch number
      case 1
        n=n-1;
      case 2
        m=m-1;
      case 3
        n=n-1;
        m=m-1;
      end
  end
    k=k+1;
    w=cat(1,w,[n,m]);
end