import admm
import numpy as np


def estimate_background(data, max_iter=100, reltol=1e-4, lam=100.0, rho=80.0):
    s = admm.State(data, padding=10)
    it = 0
    while it < max_iter:
        it += 1
        xold = s.x
        admm.update_x(s)
        admm.update_z1(s, rho)
        admm.update_z2(s, rho, lam)
        admm.update_residuals(s)
        r = admm.compute_primal_residual(s)
        s = admm.compute_dual_residual(s, rho)
        dx = np.linalg.norm(s.x - xold) / np.linalg.norm(xold)
        print(
            f"\r iter : {it:d}, |r| = {r:10.4e}, |s| = {s:10.4e}, |dx| = {dx:10.4e}",
            end=""
        )
        if dx < reltol:
            break
    print("")

    return admm.remove_pad(s.x, x.pad)
