add_executable(find_keypoints find_keypoints.cpp)

target_include_directories(find_keypoints PRIVATE ../src)

target_link_libraries(find_keypoints PRIVATE
                      img
                      sift
)

add_executable(match_features match_features.cpp)

target_include_directories(match_features PRIVATE ../src)

target_link_libraries(match_features PRIVATE
                      img
                      sift
)

SET_TARGET_PROPERTIES(find_keypoints PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
SET_TARGET_PROPERTIES(match_features PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)