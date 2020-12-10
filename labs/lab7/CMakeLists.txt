# set the project name and version
project(main)

set(CMAKE_CXX_COMPILER  /usr/bin/mpic++)

# add the executable
add_executable(main samples/sample_mpi.cpp)