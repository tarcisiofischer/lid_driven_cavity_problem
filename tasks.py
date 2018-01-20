from invoke import task

@task
def compile(ctx):
    commands = [
        'python setup.py build_ext --inplace',
        'mkdir -p build',
        'cd build',
        'cmake ..',
        'make',
        'make install'
    ]
    ctx.run(' && '.join(commands))


@task
def cdt(ctx, build_type='Debug', target='../lid_driven_cavity_problem_cdt'):
    import os
    current_dir = os.getcwd()

    commands = [
        'mkdir -p %s' % (target,),
        'cd %s' % (target,),
        'cmake %s -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=%s' % (current_dir, build_type,),
    ]
    ctx.run(' && '.join(commands))
