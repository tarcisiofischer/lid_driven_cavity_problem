# encoding: UTF-8
from __future__ import absolute_import, division, print_function, unicode_literals
from petsc4py import PETSc


def test_petsc_solver():
    def f(snes, x, f):
        for i in range(N):
            f[i] = (x[i] ** 2 - 9.0).item()
        f[N//2] = x[N//2]
        f.assemble()

    COMM = PETSc.COMM_WORLD
    N = 4
    J = PETSc.Mat().createAIJ(N, comm=COMM)
    J.setPreallocationNNZ(N)
    for i in range(N):
        for j in range(N):
            J.setValue(i, j, 0.0)
    J.setUp()
    J.assemble()

    dm = PETSc.DMShell().create(comm=COMM)
    dm.setMatrix(J)

    snes = PETSc.SNES().create(comm=COMM)
    r = PETSc.Vec().createSeq(N)
    x = PETSc.Vec().createSeq(N)
    b = PETSc.Vec().createSeq(N)
    snes.setFunction(f, r)
    snes.setDM(dm)

    # This enables coloring
    snes.setUseFD(True)

    x.setArray([1.1] * N)
    b.set(0)

    snes.setConvergenceHistory()
    snes.setFromOptions()
    snes.solve(b, x)

    REASONS = {
        0: 'still iterating',
        # Converged
        2: '||F|| < atol',
        3: '||F|| < rtol',
        4: 'Newton computed step size small; || delta x || < stol || x ||',
        5: 'maximum iterations reached',
        7: 'trust region delta',
        # Diverged
        -1: 'the new x location passed to the function is not in the domain of F',
        -2: 'maximum function count reached',
        -3: 'the linear solve failed',
        -4: 'norm of F is NaN',
        - 5: 'maximum iterations reached',
        -6: 'the line search failed',
        -7: 'inner solve failed',
        -8: '|| J^T b || is small, implies converged to local minimum of F()',
    }
    print("")
    print("Diverged with reason:", REASONS[snes.reason])
    print(x.getArray())
