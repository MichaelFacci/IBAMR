## ---------------------------------------------------------------------
##
## Copyright (c) 2020 - 2021 by the IBAMR developers
## All rights reserved.
##
## This file is part of IBAMR.
##
## IBAMR is free software and is distributed under the 3-clause BSD
## license. The full text of the license can be found in the file
## COPYRIGHT at the top level directory of IBAMR.
##
## ---------------------------------------------------------------------

IBAMR_ADD_EXAMPLE(
  TARGET_NAME
    "IIM-ex3-2d"
  OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/examples/IIM/ex3"
  OUTPUT_NAME
    main2d
  EXAMPLE_GROUP
    examples-IIM
  SOURCES
    example.cpp
  REQUIRES
    IBAMR_HAVE_LIBMESH
  LINK_TARGETS
    IBAMR2d
  INPUT_FILES
    input2d
  EXTRA_FILES
    compute_St.m
    airfoil_bc_0.1_TRI3_0.1.e
    )
