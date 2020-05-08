function [C,U,R]  = CUR_article(A,c)
	Px = [] ;
	Qx = [] ;
	[m,n] = size(A) ;
	for x =  1:n 
		Px = [ Px,norm(A(:,x),2) / norm(A) ]; 
	
	end
	C_i = randsample(1:n,c,true,Px);
	C = A(:,C_i);

	for x =  1:m 
		Qx = [ Qx,norm(A(x,:),2) / norm(A) ]; 
	
	end
	
	R_i = randsample(1:m,c,true,Qx);
	R = A(R_i,:);
	
	U = pinv(C) * A * pinv(R);

end