load Af.mat;
load Ac.mat;
load P.mat;


%spy(Af)
%spy(Ac)
spy(P)

return;

trace(inv(full(Af)))

%P

%return;

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