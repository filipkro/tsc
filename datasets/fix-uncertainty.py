import numpy as np
from argparse import ArgumentParser


femval_cert = {'1':1, '4':2, '8':2, '13':1, '14':1,'19':1, '23':1, '25':1, '42':1, '45':2, '52':1, '54':1, '67':2, '73':1, '74':2, '78':1, '86':2, '90':1, '92':1, '93':1, '97':2, '103':1}
trunk_cert = {'1':1, '4':3, '8':2, '13':1, '14':1,'19':2, '23':1, '25':1, '42':1, '45':2, '52':2, '54':1, '67':1, '73':1, '74':1, '78':2, '86':1, '90':1, '92':1, '93':1, '97':2, '103':1}
hip = {'1':1, '4':2, '8':1, '13':1, '14':1,'19':2, '23':1, '25':1, '42':2, '45':1, '52':1, '54':1, '67':1, '73':1, '74':2, '78':2, '86':1, '90':1, '92':2, '93':2, '97':3, '103':1}

kmfp_cert = {'1':1, '4':1, '8':1, '13':1, '14':3,'19':1, '23':1, '25':2, '42':1, '45':1, '52':1, '54':1, '67':1, '73':1, '74':1, '78':1, '86':1, '90':2, '92':1, '93':1, '97':1, '103':1}
# fuck = {'49':1}

def main(args):
