function A_sampled = ColumnSelect(A, k , eps)
	A_cur =[];
	J=[];
	C = k*log(k/(eps^2)) ;
	%C = k ;
	[m,n] = size(A) ; 
	%[u,sigma , v] = svd(A) 
	%[W,lambda] = eig(A' * A) ; 
	%W = W(:, end:-1:1) ;  
	%lambda = lambda(:, end:-1:1) ;
	[UU,SS,W ] =  svd(A);
	

	W_top = W(:,1:k)	;
	for j = 1:n
		pie = norm(W_top(j,:),2)^2/k ;
		pj  = min(1,C*pie) ;
		prob = rand ;
		if (prob <= pj)   
			A_cur = [A_cur,A(:,j) ] ; 
			J=[J,j];
		end
		
		
	end
	A_sampled = A_cur ;
	size(J)
end