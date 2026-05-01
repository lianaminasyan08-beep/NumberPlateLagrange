import ctypes
import numpy as np
import os

lib = ctypes.CDLL("./cpp_module/liblagrange.dll")

# encode
lib.encode_plate.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

# signature
lib.build_signature.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

# compare
lib.compare.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.compare.restype = ctypes.c_double


def encode_plate_cpp(text):
    arr = (ctypes.c_int * 100)()
    size = ctypes.c_int()

    lib.encode_plate(text.encode(), arr, ctypes.byref(size))

    return list(arr[:size.value])


def build_signature_cpp(nums):
    arr_in = (ctypes.c_int * len(nums))(*nums)
    arr_out = (ctypes.c_double * 60)()

    lib.build_signature(arr_in, len(nums), arr_out)

    return np.array(arr_out)


def compare_cpp(a, b):
    a_arr = (ctypes.c_double * len(a))(*a)
    b_arr = (ctypes.c_double * len(b))(*b)

    return lib.compare(a_arr, b_arr, len(a))