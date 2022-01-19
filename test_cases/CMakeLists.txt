add_app(
    APP_NAME case_sh
    APP_SRCS case_img_sum_shmem.cu
)

add_app(
    APP_NAME case_tex
    APP_SRCS case_img_sum_texture.cu
)

add_app(
    APP_NAME case_atom
    APP_SRCS case_atomics.cu
)

