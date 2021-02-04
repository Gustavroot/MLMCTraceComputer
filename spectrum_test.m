load Af.mat;
load Ac.mat;
load P.mat;

%spy(Af)
%spy(Ac)
%spy(P)

Qf = Af(:,:);
[m,n] = size(Qf);
half_size = int32(m/2);
Qf(half_size+1:half_size*2,:) = -Qf(half_size+1:half_size*2,:);

trQ = trace(inv(full(Qf)));
trAf = trace(inv(full(Af)));

display(trQ);
display(trAf);

diffD = inv(full(Af)) - P*(inv(full(Ac)))*P';

off_diffD = diffD - diag(diag(diffD));

norm(off_diffD,'fro')

return;

Df = eig(inv(full(Af)));
plot(Df,'o');

hold on

Dc = eig(P*inv(full(P'*Af*P))*P');
plot(Dc,'x');

hold off