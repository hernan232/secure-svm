svm = fxp_svm(2, 0.01);

X = [1, 2; 2, 1];
y = [1; -1];

disp(size(svm.W))

svm = svm.fit(X, y, 10);

disp(size(svm.W))

svm.score(X, y)