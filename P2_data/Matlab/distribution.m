function out=distribution(m,v,g,x,y)  % This function I haven't checked it, manuel to alex, 17/10/21
for i=1:size(m,1)
    dx = x-m(i,1);
    dy = x-m(i,2);
    amp = g(i,:)/sqrt(2*pi*v(i,:));
    out(:,:,i) = amp*exp(-0.5 * (dx.*dx)/v(i,1)+ (dy.*dy)/v(i,2));
end
end
