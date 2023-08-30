import admm
import numpy as np


def estimate_background(data, max_iter=100, reltol=1e-4, lam=100.0, rho=80.0):
    s = admm.State(data, padding=10)
    it = 0
    primal_residuals = []
    dual_residuals = []
    dx_list = []

    while it < max_iter:
        it += 1
        xold = s.x
        admm.update_x(s)
        admm.update_z1(s, rho)
        admm.update_z2(s, rho, lam)
        admm.update_residuals(s)
        primal_resid = admm.compute_primal_residual(s)
        dual_resid = admm.compute_dual_residual(s, rho)
        dx = np.linalg.norm(s.x - xold) / max(np.linalg.norm(xold), 1e-8)
        primal_residuals.append(primal_resid)
        dual_residuals.append(dual_resid)
        dx_list.append(dx)
        print(
            f"\r iter : {it:d}, |r| = {primal_resid:10.4e}, |s| = {dual_resid:10.4e}, |dx| = {dx:10.4e}",
            end="",
        )
        if dx < reltol:
            break
    print("")

    return admm.remove_pad(s.x, s.pad) * s.scale, {
        "primal": primal_residuals,
        "dual": dual_residuals,
        "dx": dx_list,
    }
