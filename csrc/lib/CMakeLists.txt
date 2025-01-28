set(LIB_NAME "torchOp")

file(GLOB_RECURSE CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

Python_add_library(${LIB_NAME} SHARED ${CXX_SOURCES} ${CUDA_SOURCES})
add_library(${PROJECT_NAMESPACE}${LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(
    ${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    ${LIB_NAME} 
    PUBLIC 
    ${TORCH_LIBRARIES}
)

install_libraries(
    ${LIB_NAME}
    ${PROJECT_NAMESPACE}
    ${LIB_NAME}
)