# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/jyao073/miniconda3/envs/CrowdNav/lib/python3.11/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/jyao073/miniconda3/envs/CrowdNav/lib/python3.11/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jyao073/CP_Safe_Meta_RL/Python-RVO2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2

# Include any dependencies generated for this target.
include examples/CMakeFiles/Blocks.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/Blocks.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/Blocks.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/Blocks.dir/flags.make

examples/CMakeFiles/Blocks.dir/Blocks.cpp.o: examples/CMakeFiles/Blocks.dir/flags.make
examples/CMakeFiles/Blocks.dir/Blocks.cpp.o: /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/examples/Blocks.cpp
examples/CMakeFiles/Blocks.dir/Blocks.cpp.o: examples/CMakeFiles/Blocks.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/Blocks.dir/Blocks.cpp.o"
	cd /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/Blocks.dir/Blocks.cpp.o -MF CMakeFiles/Blocks.dir/Blocks.cpp.o.d -o CMakeFiles/Blocks.dir/Blocks.cpp.o -c /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/examples/Blocks.cpp

examples/CMakeFiles/Blocks.dir/Blocks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Blocks.dir/Blocks.cpp.i"
	cd /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/examples/Blocks.cpp > CMakeFiles/Blocks.dir/Blocks.cpp.i

examples/CMakeFiles/Blocks.dir/Blocks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Blocks.dir/Blocks.cpp.s"
	cd /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/examples/Blocks.cpp -o CMakeFiles/Blocks.dir/Blocks.cpp.s

# Object files for target Blocks
Blocks_OBJECTS = \
"CMakeFiles/Blocks.dir/Blocks.cpp.o"

# External object files for target Blocks
Blocks_EXTERNAL_OBJECTS =

examples/Blocks: examples/CMakeFiles/Blocks.dir/Blocks.cpp.o
examples/Blocks: examples/CMakeFiles/Blocks.dir/build.make
examples/Blocks: src/libRVO.a
examples/Blocks: examples/CMakeFiles/Blocks.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Blocks"
	cd /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Blocks.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/Blocks.dir/build: examples/Blocks
.PHONY : examples/CMakeFiles/Blocks.dir/build

examples/CMakeFiles/Blocks.dir/clean:
	cd /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/examples && $(CMAKE_COMMAND) -P CMakeFiles/Blocks.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/Blocks.dir/clean

examples/CMakeFiles/Blocks.dir/depend:
	cd /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jyao073/CP_Safe_Meta_RL/Python-RVO2 /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/examples /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2 /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/examples /home/jyao073/CP_Safe_Meta_RL/Python-RVO2/build/RVO2/examples/CMakeFiles/Blocks.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : examples/CMakeFiles/Blocks.dir/depend

