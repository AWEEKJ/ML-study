# Lecture 2: Linear Regression

- 이미 존재하는 데이터(training data set)로 Regression model을 학습시킴(training)
- 세상의 많은 현상들이 리니어한 모델로 설명할 수 있다! 
- (Linear) Hypothesis: H(x) = Wx+b
- Which hypothesis is better? => Cost Function(Loss Function)
- cost(W, b) = average of (H(x)-y)^2
- Goal: Minimize cost(W, b)