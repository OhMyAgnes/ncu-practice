# A helper function for adding applications for libraries.
function(add_app)
    cmake_parse_arguments(
        PARSED_ARGS           # prefix of output variables
        ""                    # list of names of the boolean arguments (only defined ones will be true)
        "APP_NAME"            # list of names of mono-valued arguments
        "APP_SRCS;APP_DEPS"   # list of names of multi-valued arguments (output variables are lists)
        ${ARGN}               # arguments of the function to parse, here we take the all original ones
    )
    add_executable(${PARSED_ARGS_APP_NAME})

    target_sources(${PARSED_ARGS_APP_NAME}
        PRIVATE 
            ${PARSED_ARGS_APP_SRCS}
    )

    target_include_directories(${PARSED_ARGS_APP_NAME}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(${PARSED_ARGS_APP_NAME} 
        PRIVATE 
            ${PARSED_ARGS_APP_DEPS}
    )

endfunction(add_app)