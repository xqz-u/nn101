cmake_minimum_required(VERSION 3.12)
project(nn101)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wunused
  -Wconversion
  -Wfloat-equal
  -Waggregate-return
  -g
)

# 3. Find GSL automatically
# This looks in standard system paths (Homebrew, /usr/lib, etc.)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSL REQUIRED gsl)

# Automatically grab all .c files in src/
file(GLOB SOURCES "src/*.c")

add_executable(a.out ${SOURCES})

# 5. Link GSL
# Include directories and link libraries
target_include_directories(a.out PRIVATE ${GSL_INCLUDE_DIRS})
target_link_directories(a.out PRIVATE ${GSL_LIBRARY_DIRS})
target_link_libraries(a.out PRIVATE ${GSL_LIBRARIES})
