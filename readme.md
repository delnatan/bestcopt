# Flexible sheet simulation for background estimation using convex optimization

This repo is a work-in-progress for a background subtraction algorithm using convex optimization. The problem statement is:

$$
\begin{aligned}
\underset{x}{\arg\min} \quad & \|b - x\|_1 + \lambda \|\mathbf{D}x\|_2^2\\
\text{subject to}\quad & x \leq b
\end{aligned}
$$

Where $b$ is data and $x$ is the baseline to be solved. The matrix $\mathbf{D}$ is a finite-difference linear operator controlling for 'pliability' or smoothness of $x$, controlled by the scalar $\lambda$. The popular algorithm ADMM (Alternating direction method of multiplier) is used to solve the problem. $x$ can be 1, 2, or 3-dimensional.

more details to come ...
