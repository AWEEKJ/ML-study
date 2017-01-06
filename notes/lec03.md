# Lecture 3: How to minimize cost

## Simplified Hypothesis

## Gradient descent algorithm

- 경사를 따라 내려가는 알고리즘
- Minimize cost function
- Gradient descent is used many minimization problems
- For a given cost function, cost(W, b), it will find W, b to minimize cost
- It can be applied to more general function: cost(W1, W2, ...)

### How it works? How would you find thd lowest point?

- Start with initial guesses
	- Start at (0,0) (or any other value) *아무 점에서 시작한다.*
	- Keeping changing W and b a little bit to try and reduce cost(W, b) 
- Each time you change the parameters, you select the gradient which reduces cost(W, b) the most possible
- Repeat
- Do so until you converge to a local minimum
- Has an interesting property
	- Where you start can determine which minimum you end up
- 기울기는 미분해서 구한다. 기울기 < 0 이면 W를 크게, 기울기 > 0 이면 W를 작게 만든다.
- cost(W, b)를 설계할 때 그 모양이 Convex function인지를 확인하라.
- Convex function: 밥그릇 모양. Gradient descent algorithm을 사용할 때, 초기값으로 무엇을 주더라도 같은 결과가 나온다.
