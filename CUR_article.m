function [C,U,R]  = CUR_article(A,k,eps)
	%disp('A') ;
	%size(A)
	C = ColumnSelect(A,k,eps) ;
	%disp('C') ;
	%size((C))
	%disp('pinvC') ;
	%size(pinv(C))
	R_x = ColumnSelect(A',k,eps) ;
	R = R_x';
	%disp('R') ;
	%size((R))
	%disp('pinvR') ;
	%size(pinv(R))
	U = pinv(C) * A * pinv(R) ;
	%disp('U') ;
	%size(U)
end