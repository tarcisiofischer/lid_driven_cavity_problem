from petsc4py import PETSc

COMM = PETSc.COMM_WORLD

dm = PETSc.DMShell().create(comm=COMM)

N = 2

mat = PETSc.Mat().createAIJ(N, comm=COMM)

# could also use `create` and then setSizes below:
# mat.setSizes(((2, None), (2, None)))

mat.setPreallocationNNZ(2)

# There must be a better way to fill the matrix structure...
# (If these setValues are not done, it will segfault)
for i in range(N):
    for j in range(N):
        mat.setValue(i, j, 0)

mat.setUp()
mat.assemble()

# print(mat.getValuesCSR())

dm.setMatrix(mat)


class Function:
    def __call__(self, snes, x, f):
        f[0] = (x[0] * x[0] + x[0] * x[1] - 3.0).item()
        f[1] = (x[0] * x[1] + x[1] * x[1] - 6.0).item()
        f.assemble()


snes = PETSc.SNES().create(comm=COMM)

r = PETSc.Vec().createSeq(N)
x = PETSc.Vec().createSeq(N)
b = PETSc.Vec().createSeq(N)

snes.setFunction(Function(), r)


snes.setDM(dm)
snes.setUseFD(True)

x.setArray([2, 3])
b.set(0)

snes.setConvergenceHistory()
snes.setFromOptions()
snes.solve(b, x)
rh, ih = snes.getConvergenceHistory()

print(list(zip(rh, ih)))
print(x.getArray())
