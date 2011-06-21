import os
import sconsconfig as config
from sconsconfig import packages, tools

# Set the project name.
proj_name = 'gpu-nbody'

#
# Select the packages we want to use in the configuration.
#

# config.select(
#     packages.MPI(required=True),
#     packages.GSL(required=True),
# )

#
# Setup the variables available for this build.
#

vars = Variables('config.py') # Persistent storage.
vars.AddVariables(
    ('CC', 'Set C compiler.'),
    ('CXX', 'Set CXX compiler.'),
    EnumVariable('BUILD', 'Set the build type.', 'debug', allowed_values=('debug', 'optimised')),
    EnumVariable('BITS', 'Set number of bits.', '64', allowed_values=('32', '64')),
    BoolVariable('PROF', 'Enable profiling.', False),
    BoolVariable('WITH_TAU', 'Enable tau profiling.', False),
    BoolVariable('WITH_GCOV', 'Enable coverage testing with gcov.', False),
    PathVariable('PREFIX', 'Set install location.', '/usr/local', PathVariable.PathIsDirCreate),
    BoolVariable('BUILD_STATIC_LIBS', 'Build static libraries.', True),
    BoolVariable('BUILD_SHARED_LIBS', 'Build shared libraries.', True),
    BoolVariable('BUILD_TESTS', 'Build unit tests.', True),
    BoolVariable('BUILD_APPS', 'Build applications.', True),
    BoolVariable('BUILD_EXS', 'Build applications.', True),
    BoolVariable('BUILD_DOC', 'Build documentation.', False),
)

# Add options from any packages we want to use.
config.add_options(vars)

#
# Create the construction environment we will use.
#

# Build a list of tools we want included.
tools = ['default', 'cuda', 'cxxtest']

# Create the environment.
env = Environment(tools=tools, toolpath=[config.__path__[0] + '/tools'], variables=vars,
                  ENV=os.environ, CUDA_TOOLKIT_PATH='/opt/local/cuda', CUDA_SDK_PATH='/opt/local/cuda')

# Add any custom tools here.

# Check if there were any unkown variables on the command line.
unknown = vars.UnknownVariables()
if unknown:
    print 'Unknown variables:', unknown.keys()
    env.Exit(1)

# Take a snapshot of provided options before we continue.
vars.Save('config.py', env)

# Generate a help line later use.
Help(vars.GenerateHelpText(env))

# If the user requested help don't bother continuing with the build.
if not GetOption('help'):

    #
    # Perform configuration of the project.
    #

    # Create our configuration environment, passing the set of custom tests.
    sconf = env.Configure(custom_tests=config.custom_tests)

    # Run our custom tests with any options needed.
    config.check(sconf)

    # Finish the configuration and save it to file.
    sconf.Finish()
    vars.Save('config.py', env)

    # Modify the environment based on any of our variables.
    if env['BUILD'] == 'debug':
        env.MergeFlags('-g -O0')
        env.AppendUnique(NVCCFLAGS=['-g'])
    elif env['BUILD'] == 'optimised':
        env.MergeFlags('-DNDEBUG -O3')
    if env['BITS'] == '64':
        env.MergeFlags('-m64')
        env.AppendUnique(NVCCFLAGS=['-m64'])
        env.AppendUnique(LINKFLAGS=['-m64'])
    else:
        env.MergeFlags('-m32')
        env.AppendUnique(NVCCFLAGS=['-m32'])
        env.AppendUnique(LINKFLAGS=['-m32'])
    if env['PROF']:
        env.MergeFlags('-g -pg')
        env.AppendUnique(LINKFLAGS=['-pg'])
    if env['WITH_GCOV']:
        env.AppendUnique(CFLAGS=['-fprofile-arcs', '-ftest-coverage'])
        env.AppendUnique(CCFLAGS=['-fprofile-arcs', '-ftest-coverage'])
        env.AppendUnique(LINKFLAGS=['-fprofile-arcs', '-ftest-coverage'])
    if env['WITH_TAU']:
        env['CC'] = 'tau_cc.sh'
        env['CXX'] = 'tau_cxx.sh'
        env.AppendUnique(CPPDEFINES=['WITH_TAU'])
        env.AppendUnique(CPPDEFINES=['NDEBUG'])

    # Make sure we can find CxxTest.
    env.PrependUnique(CPPPATH=['#cxxtest/include'])

    # Make sure our source code can locate installed headers and
    # libraries.
    env['BUILD'] = 'build-' + env['BUILD']
    env.PrependUnique(CPPPATH=[
        '#' + env['BUILD'] + '/include',
        '#' + env['BUILD'] + '/include/' + proj_name,
    ])
    env.PrependUnique(LIBPATH=['#' + env['BUILD'] + '/lib'])

    #
    # Begin specifying targets.
    #

    # Call sub scripts.
    Export('env')
    SConscript(proj_name + '/SConscript', variant_dir=env['BUILD'] + '/' + proj_name, duplicate=0)

    #
    # Alias any special targets.
    #

    env.Alias('install', env['PREFIX'])
