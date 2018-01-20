from invoke import task

@task
def compile(ctx):
    ctx.run("python setup.py build_ext --inplace")
    ctx.run("mkdir -p build")
    ctx.run("cd build && cmake .. && make && make install")
