%ColumnSelect([1 2 3 ; 4 5 6 ; 7 8 10], 2 , 2)
filename = 'patches.mat';
%myVars = {'X','caption'};
 load(filename)
 %figure ;
 X= patches(:,1:595)' ; 
 %imagesc(reshape(patches(:,1),20,20)) ; colormap gray ;
 %ColumnSelect(patches',11, 1) ; % 11-->21 , 18---> 51  
 %[C,U,R] =CUR_article(X,8,.1) ; % (5,.1) --> c=25 r=31       \\\\  8,.1 --- c = 52  r = 54 
 [C, U,R] = CUR(X , 2) ;
Prj = C *U ;

norm (X - C*U*R)  
%size(C*U*R) ;