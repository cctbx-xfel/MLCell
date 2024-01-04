import glob
import logging
import os
import re
import requests
import scipy.stats as stats
import sys
import time
import traceback
import yaml

import numpy as np
import pandas as pd

def calculate_g6(reduced_uc_params):
    mm11 = np.square(reduced_uc_params[0])
    mm22 = np.square(reduced_uc_params[1])
    mm33 = np.square(reduced_uc_params[2])
    mm12 = 2*reduced_uc_params[1]*reduced_uc_params[2]*np.cos(np.radians(reduced_uc_params[3]))
    mm23 = 2*reduced_uc_params[0]*reduced_uc_params[2]*np.cos(np.radians(reduced_uc_params[4]))
    mm31 = 2*reduced_uc_params[0]*reduced_uc_params[1]*np.cos(np.radians(reduced_uc_params[5]))
    return [mm11, mm22, mm33, mm12, mm23, mm31]

def calculate_s6(g6_params):
    # [P, Q, R, S, T, U] = [s23, s13, s12, s14, s24, s34] = [bdotc, adotc, adotb, adotd, bdotd, cdotd]
    tol = 1e-10
    p = g6_params[3]/2.0
    q = g6_params[4]/2.0
    r = g6_params[5]/2.0
    s = (-2.0*g6_params[0] - g6_params[5] - g6_params[4]) / 2.0
    t = (-1.0*g6_params[5] - 2.0*g6_params[1] - g6_params[3]) / 2.0
    u = (-1.0*g6_params[4] - g6_params[3] - 2.0*g6_params[2]) / 2.0
    #tselling_vector = np.array([np.dot(b,c), np.dot(a,c), np.dot(a,b),
    #                           np.dot(a,d), np.dot(b,d), np.dot(c,d),
    #                          ]
    #                             )
    selling_vector = np.array([p, q, r, s, t, u])
    selling_vector = np.array([s if abs(s) > tol else 0 for s in selling_vector])
    #print(tselling_vector,selling_vector)
    reduction_matrices = np.array(
                                      [
                                       np.array(
                                                [
                                                 [-1, 0, 0, 0, 0, 0],
                                                 [ 1, 1, 0, 0, 0, 0],
                                                 [ 1, 0, 0, 0, 1, 0],
                                                 [-1, 0, 0, 1, 0, 0],
                                                 [ 1, 0, 1, 0, 0, 0],
                                                 [ 1, 0, 0, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 1, 0, 0, 0, 0],
                                                 [ 0,-1, 0, 0, 0, 0],
                                                 [ 0, 1, 0, 1, 0, 0],
                                                 [ 0, 1, 1, 0, 0, 0],
                                                 [ 0,-1, 0, 0, 1, 0],
                                                 [ 0, 1, 0, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 0, 1, 0, 0, 0],
                                                 [ 0, 0, 1, 1, 0, 0],
                                                 [ 0, 0,-1, 0, 0, 0],
                                                 [ 0, 1, 1, 0, 0, 0],
                                                 [ 0, 0, 1, 0, 1, 0],
                                                 [ 0, 0,-1, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 0, 0,-1, 0, 0],
                                                 [ 0, 0, 1, 1, 0, 0],
                                                 [ 0, 1, 0, 1, 0, 0],
                                                 [ 0, 0, 0,-1, 0, 0],
                                                 [ 0, 0, 0, 1, 1, 0],
                                                 [ 0, 0, 0, 1, 0, 1],
                                                ]
                                               ),
                                     np.array(
                                              [
                                               [ 0, 0, 1, 0, 1, 0],
                                               [ 0, 1, 0, 0,-1, 0],
                                               [ 1, 0, 0, 0, 1, 0],
                                               [ 0, 0, 0, 1, 1, 0],
                                               [ 0, 0, 0, 0,-1, 0],
                                               [ 0, 0, 0, 0, 1, 1],
                                              ]
                                             ),
                                     np.array(
                                              [
                                               [ 0, 1, 0, 0, 0, 1],
                                               [ 1, 0, 0, 0, 0, 1],
                                               [ 0, 0, 1, 0, 0,-1],
                                               [ 0, 0, 0, 1, 0, 1],
                                               [ 0, 0, 0, 0, 1, 1],
                                               [ 0, 0, 0, 0, 0,-1],
                                              ]
                                             ),
                                  ]
                                 )

    while np.greater(np.max(selling_vector), 0):
        max_index = selling_vector.argmax()
        selling_vector = np.dot(reduction_matrices[max_index], selling_vector)

    if np.max(selling_vector) > 0:
        print("Selling vector reduction failed")    
        return None
    else:
        return selling_vector

def selling_distance(svector1, svector2):
    vcp_transform_mats = [
         np.array(
                  [
                       [-1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0,-1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0,],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0,-1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0,-1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0,-1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ), 
             np.array(
                      [
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0 ,0 ,0 ,0, 0,-1],
                  ]
                 ),
        ] 


    reflection_mats = [
         np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0 ,0 ,0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0 ,0 ,0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1 ,0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
                        
             np.array(
                      [ 
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                      ]
                     ),
             np.array( #10
                      [ 
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                      ]
                     ),
             np.array(#20
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                      ]
                 ),
        ]

    vcps = np.dot(svector1, vcp_transform_mats)[0]

    all_reflections = []
    for vcp in vcps:
        for mat in reflection_mats:
            all_reflections.append(np.dot(vcp, mat))
    for mat in reflection_mats:
        all_reflections.append(np.dot(svector1, reflection_mat))
         
    return np.min(np.linalg.norm(reflection-svector2) for reflection in all_reflections)

def calculate_uc_from_g6(g6_params):
    g1 = g6_params[0]
    g2 = g6_params[1]
    g3 = g6_params[2]
    g4 = g6_params[3]
    g5 = g6_params[4]
    g6 = g6_params[5]

    a = np.sqrt(g1)
    b = np.sqrt(g2)
    c = np.sqrt(g3)
    alpha = (np.arccos(g4/(2*b*c)))*(180.0/np.pi)
    beta = (np.arccos(g5/(2*a*c)))*(180.0/np.pi)
    gamma = (np.arccos(g6/(2*a*b)))*(180.0/np.pi)
    return [a, b, c, alpha, beta, gamma]

def calculate_uc_from_s6(s6_params):
    print('s6',len(s6_params),s6_params)
    p = s6_params[0]
    q = s6_params[1]
    r = s6_params[2]
    s = s6_params[3]
    t = s6_params[4]
    u = s6_params[5]
    g1 = -1*q-r-s
    g2 = -1*p-r-t
    g3 = -1*p-q-u
    g4 = 2*p
    g5 = 2*q
    g6 = 2*r
    return calculate_uc_from_g6([g1,g2,g3,g4,g5,g6])

