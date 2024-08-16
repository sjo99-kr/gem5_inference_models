cmake_minimum_required(VERSION 3.10)

# 프로젝트 이름 설정
project(my_app_resnet)

# GEM5_HOME 설정
set(GEM5_HOME /home/airplon/screen/gem5)

# OpenCV 찾기
find_package(OpenCV REQUIRED)

# Include 디렉토리 설정
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GEM5_HOME}/include)

# 링크할 라이브러리 디렉토리 설정
link_directories(${GEM5_HOME}/util/m5/build/x86/out)

# C++ 컴파일러 플래그 설정
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# 실행 파일 생성
add_executable(my_app_resnet main.cpp)

# OpenCV 및 GEM5의 m5 라이브러리 링크
target_link_libraries(my_app_resnet ${OpenCV_LIBS} m5)
