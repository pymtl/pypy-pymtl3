# THIS FILE IS AUTOMATICALLY GENERATED BY gendfa.py
# DO NOT EDIT
# TO REGENERATE THE FILE, RUN:
#     python gendfa.py > dfa_generated.py

from pypy.interpreter.pyparser import automata
accepts = [True, True, True, True, True, True, True, True,
           True, True, True, False, True, True, True, True,
           True, False, False, False, False, True, False,
           False, False, True, False, False, True, False,
           False, True, False, True, False, True, False,
           False, True, False, False, True, False, True,
           False, True, False, True, False, False, True,
           False, False, True, True, False, False, False,
           False, True, False, True, False, True, False,
           False, True, False, True, True, False, True,
           False, True, False, True, False, True, True, True,
           True, True]
states = [
    # 0
    {'\t': 0, '\n': 15, '\x0c': 0,
     '\r': 16, ' ': 0, '!': 11, '"': 19,
     '#': 21, '$': 17, '%': 14, '&': 14,
     "'": 18, '(': 15, ')': 15, '*': 8,
     '+': 14, ',': 15, '-': 12, '.': 7,
     '/': 13, '0': 5, '1': 6, '2': 6,
     '3': 6, '4': 6, '5': 6, '6': 6,
     '7': 6, '8': 6, '9': 6, ':': 15,
     ';': 15, '<': 10, '=': 14, '>': 9,
     '@': 14, 'A': 1, 'B': 2, 'C': 1,
     'D': 1, 'E': 1, 'F': 2, 'G': 1,
     'H': 1, 'I': 1, 'J': 1, 'K': 1,
     'L': 1, 'M': 1, 'N': 1, 'O': 1,
     'P': 1, 'Q': 1, 'R': 3, 'S': 1,
     'T': 1, 'U': 4, 'V': 1, 'W': 1,
     'X': 1, 'Y': 1, 'Z': 1, '[': 15,
     '\\': 20, ']': 15, '^': 14, '_': 1,
     '`': 15, 'a': 1, 'b': 2, 'c': 1,
     'd': 1, 'e': 1, 'f': 2, 'g': 1,
     'h': 1, 'i': 1, 'j': 1, 'k': 1,
     'l': 1, 'm': 1, 'n': 1, 'o': 1,
     'p': 1, 'q': 1, 'r': 3, 's': 1,
     't': 1, 'u': 4, 'v': 1, 'w': 1,
     'x': 1, 'y': 1, 'z': 1, '{': 15,
     '|': 14, '}': 15, '~': 15,
     '\x80': 1},
    # 1
    {'0': 1, '1': 1, '2': 1, '3': 1,
     '4': 1, '5': 1, '6': 1, '7': 1,
     '8': 1, '9': 1, 'A': 1, 'B': 1,
     'C': 1, 'D': 1, 'E': 1, 'F': 1,
     'G': 1, 'H': 1, 'I': 1, 'J': 1,
     'K': 1, 'L': 1, 'M': 1, 'N': 1,
     'O': 1, 'P': 1, 'Q': 1, 'R': 1,
     'S': 1, 'T': 1, 'U': 1, 'V': 1,
     'W': 1, 'X': 1, 'Y': 1, 'Z': 1,
     '_': 1, 'a': 1, 'b': 1, 'c': 1,
     'd': 1, 'e': 1, 'f': 1, 'g': 1,
     'h': 1, 'i': 1, 'j': 1, 'k': 1,
     'l': 1, 'm': 1, 'n': 1, 'o': 1,
     'p': 1, 'q': 1, 'r': 1, 's': 1,
     't': 1, 'u': 1, 'v': 1, 'w': 1,
     'x': 1, 'y': 1, 'z': 1, '\x80': 1},
    # 2
    {'"': 19, "'": 18, '0': 1, '1': 1,
     '2': 1, '3': 1, '4': 1, '5': 1,
     '6': 1, '7': 1, '8': 1, '9': 1,
     'A': 1, 'B': 1, 'C': 1, 'D': 1,
     'E': 1, 'F': 1, 'G': 1, 'H': 1,
     'I': 1, 'J': 1, 'K': 1, 'L': 1,
     'M': 1, 'N': 1, 'O': 1, 'P': 1,
     'Q': 1, 'R': 4, 'S': 1, 'T': 1,
     'U': 1, 'V': 1, 'W': 1, 'X': 1,
     'Y': 1, 'Z': 1, '_': 1, 'a': 1,
     'b': 1, 'c': 1, 'd': 1, 'e': 1,
     'f': 1, 'g': 1, 'h': 1, 'i': 1,
     'j': 1, 'k': 1, 'l': 1, 'm': 1,
     'n': 1, 'o': 1, 'p': 1, 'q': 1,
     'r': 4, 's': 1, 't': 1, 'u': 1,
     'v': 1, 'w': 1, 'x': 1, 'y': 1,
     'z': 1, '\x80': 1},
    # 3
    {'"': 19, "'": 18, '0': 1, '1': 1,
     '2': 1, '3': 1, '4': 1, '5': 1,
     '6': 1, '7': 1, '8': 1, '9': 1,
     'A': 1, 'B': 4, 'C': 1, 'D': 1,
     'E': 1, 'F': 4, 'G': 1, 'H': 1,
     'I': 1, 'J': 1, 'K': 1, 'L': 1,
     'M': 1, 'N': 1, 'O': 1, 'P': 1,
     'Q': 1, 'R': 1, 'S': 1, 'T': 1,
     'U': 1, 'V': 1, 'W': 1, 'X': 1,
     'Y': 1, 'Z': 1, '_': 1, 'a': 1,
     'b': 4, 'c': 1, 'd': 1, 'e': 1,
     'f': 4, 'g': 1, 'h': 1, 'i': 1,
     'j': 1, 'k': 1, 'l': 1, 'm': 1,
     'n': 1, 'o': 1, 'p': 1, 'q': 1,
     'r': 1, 's': 1, 't': 1, 'u': 1,
     'v': 1, 'w': 1, 'x': 1, 'y': 1,
     'z': 1, '\x80': 1},
    # 4
    {'"': 19, "'": 18, '0': 1, '1': 1,
     '2': 1, '3': 1, '4': 1, '5': 1,
     '6': 1, '7': 1, '8': 1, '9': 1,
     'A': 1, 'B': 1, 'C': 1, 'D': 1,
     'E': 1, 'F': 1, 'G': 1, 'H': 1,
     'I': 1, 'J': 1, 'K': 1, 'L': 1,
     'M': 1, 'N': 1, 'O': 1, 'P': 1,
     'Q': 1, 'R': 1, 'S': 1, 'T': 1,
     'U': 1, 'V': 1, 'W': 1, 'X': 1,
     'Y': 1, 'Z': 1, '_': 1, 'a': 1,
     'b': 1, 'c': 1, 'd': 1, 'e': 1,
     'f': 1, 'g': 1, 'h': 1, 'i': 1,
     'j': 1, 'k': 1, 'l': 1, 'm': 1,
     'n': 1, 'o': 1, 'p': 1, 'q': 1,
     'r': 1, 's': 1, 't': 1, 'u': 1,
     'v': 1, 'w': 1, 'x': 1, 'y': 1,
     'z': 1, '\x80': 1},
    # 5
    {'.': 28, '0': 25, '1': 27, '2': 27,
     '3': 27, '4': 27, '5': 27, '6': 27,
     '7': 27, '8': 27, '9': 27, 'B': 24,
     'E': 29, 'J': 15, 'O': 23, 'X': 22,
     '_': 26, 'b': 24, 'e': 29, 'j': 15,
     'o': 23, 'x': 22},
    # 6
    {'.': 28, '0': 6, '1': 6, '2': 6,
     '3': 6, '4': 6, '5': 6, '6': 6,
     '7': 6, '8': 6, '9': 6, 'E': 29,
     'J': 15, '_': 30, 'e': 29, 'j': 15},
    # 7
    {'.': 32, '0': 31, '1': 31, '2': 31,
     '3': 31, '4': 31, '5': 31, '6': 31,
     '7': 31, '8': 31, '9': 31},
    # 8
    {'*': 14, '=': 15},
    # 9
    {'=': 15, '>': 14},
    # 10
    {'<': 14, '=': 15, '>': 15},
    # 11
    {'=': 15},
    # 12
    {'=': 15, '>': 15},
    # 13
    {'/': 14, '=': 15},
    # 14
    {'=': 15},
    # 15
    {},
    # 16
    {'\n': 15},
    # 17
    {'0': 33, '1': 33, '2': 33, '3': 33,
     '4': 33, '5': 33, '6': 33, '7': 33,
     '8': 33, '9': 33},
    # 18
    {automata.DEFAULT: 37, '\n': 34,
     '\r': 34, "'": 35, '\\': 36},
    # 19
    {automata.DEFAULT: 40, '\n': 34,
     '\r': 34, '"': 38, '\\': 39},
    # 20
    {'\n': 15, '\r': 16},
    # 21
    {automata.DEFAULT: 21, '\n': 34, '\r': 34},
    # 22
    {'0': 41, '1': 41, '2': 41, '3': 41,
     '4': 41, '5': 41, '6': 41, '7': 41,
     '8': 41, '9': 41, 'A': 41, 'B': 41,
     'C': 41, 'D': 41, 'E': 41, 'F': 41,
     '_': 42, 'a': 41, 'b': 41, 'c': 41,
     'd': 41, 'e': 41, 'f': 41},
    # 23
    {'0': 43, '1': 43, '2': 43, '3': 43,
     '4': 43, '5': 43, '6': 43, '7': 43,
     '_': 44},
    # 24
    {'0': 45, '1': 45, '_': 46},
    # 25
    {'.': 28, '0': 25, '1': 27, '2': 27,
     '3': 27, '4': 27, '5': 27, '6': 27,
     '7': 27, '8': 27, '9': 27, 'E': 29,
     'J': 15, '_': 26, 'e': 29, 'j': 15},
    # 26
    {'0': 47, '1': 48, '2': 48, '3': 48,
     '4': 48, '5': 48, '6': 48, '7': 48,
     '8': 48, '9': 48},
    # 27
    {'.': 28, '0': 27, '1': 27, '2': 27,
     '3': 27, '4': 27, '5': 27, '6': 27,
     '7': 27, '8': 27, '9': 27, 'E': 29,
     'J': 15, '_': 49, 'e': 29, 'j': 15},
    # 28
    {'0': 50, '1': 50, '2': 50, '3': 50,
     '4': 50, '5': 50, '6': 50, '7': 50,
     '8': 50, '9': 50, 'E': 51, 'J': 15,
     'e': 51, 'j': 15},
    # 29
    {'+': 52, '-': 52, '0': 53, '1': 53,
     '2': 53, '3': 53, '4': 53, '5': 53,
     '6': 53, '7': 53, '8': 53, '9': 53},
    # 30
    {'0': 54, '1': 54, '2': 54, '3': 54,
     '4': 54, '5': 54, '6': 54, '7': 54,
     '8': 54, '9': 54},
    # 31
    {'0': 31, '1': 31, '2': 31, '3': 31,
     '4': 31, '5': 31, '6': 31, '7': 31,
     '8': 31, '9': 31, 'E': 51, 'J': 15,
     '_': 55, 'e': 51, 'j': 15},
    # 32
    {'.': 15},
    # 33
    {'0': 33, '1': 33, '2': 33, '3': 33,
     '4': 33, '5': 33, '6': 33, '7': 33,
     '8': 33, '9': 33},
    # 34
    {},
    # 35
    {"'": 15},
    # 36
    {automata.DEFAULT: 56, '\n': 15, '\r': 16},
    # 37
    {automata.DEFAULT: 37, '\n': 34,
     '\r': 34, "'": 15, '\\': 36},
    # 38
    {'"': 15},
    # 39
    {automata.DEFAULT: 57, '\n': 15, '\r': 16},
    # 40
    {automata.DEFAULT: 40, '\n': 34,
     '\r': 34, '"': 15, '\\': 39},
    # 41
    {'0': 41, '1': 41, '2': 41, '3': 41,
     '4': 41, '5': 41, '6': 41, '7': 41,
     '8': 41, '9': 41, 'A': 41, 'B': 41,
     'C': 41, 'D': 41, 'E': 41, 'F': 41,
     '_': 58, 'a': 41, 'b': 41, 'c': 41,
     'd': 41, 'e': 41, 'f': 41},
    # 42
    {'0': 59, '1': 59, '2': 59, '3': 59,
     '4': 59, '5': 59, '6': 59, '7': 59,
     '8': 59, '9': 59, 'A': 59, 'B': 59,
     'C': 59, 'D': 59, 'E': 59, 'F': 59,
     'a': 59, 'b': 59, 'c': 59, 'd': 59,
     'e': 59, 'f': 59},
    # 43
    {'0': 43, '1': 43, '2': 43, '3': 43,
     '4': 43, '5': 43, '6': 43, '7': 43,
     '_': 60},
    # 44
    {'0': 61, '1': 61, '2': 61, '3': 61,
     '4': 61, '5': 61, '6': 61, '7': 61},
    # 45
    {'0': 45, '1': 45, '_': 62},
    # 46
    {'0': 63, '1': 63},
    # 47
    {'.': 28, '0': 47, '1': 48, '2': 48,
     '3': 48, '4': 48, '5': 48, '6': 48,
     '7': 48, '8': 48, '9': 48, 'E': 29,
     'J': 15, '_': 26, 'e': 29, 'j': 15},
    # 48
    {'.': 28, '0': 48, '1': 48, '2': 48,
     '3': 48, '4': 48, '5': 48, '6': 48,
     '7': 48, '8': 48, '9': 48, 'E': 29,
     'J': 15, '_': 49, 'e': 29, 'j': 15},
    # 49
    {'0': 48, '1': 48, '2': 48, '3': 48,
     '4': 48, '5': 48, '6': 48, '7': 48,
     '8': 48, '9': 48},
    # 50
    {'0': 50, '1': 50, '2': 50, '3': 50,
     '4': 50, '5': 50, '6': 50, '7': 50,
     '8': 50, '9': 50, 'E': 51, 'J': 15,
     '_': 64, 'e': 51, 'j': 15},
    # 51
    {'+': 65, '-': 65, '0': 66, '1': 66,
     '2': 66, '3': 66, '4': 66, '5': 66,
     '6': 66, '7': 66, '8': 66, '9': 66},
    # 52
    {'0': 53, '1': 53, '2': 53, '3': 53,
     '4': 53, '5': 53, '6': 53, '7': 53,
     '8': 53, '9': 53},
    # 53
    {'0': 53, '1': 53, '2': 53, '3': 53,
     '4': 53, '5': 53, '6': 53, '7': 53,
     '8': 53, '9': 53, 'J': 15, '_': 67,
     'j': 15},
    # 54
    {'.': 28, '0': 54, '1': 54, '2': 54,
     '3': 54, '4': 54, '5': 54, '6': 54,
     '7': 54, '8': 54, '9': 54, 'E': 29,
     'J': 15, '_': 30, 'e': 29, 'j': 15},
    # 55
    {'0': 68, '1': 68, '2': 68, '3': 68,
     '4': 68, '5': 68, '6': 68, '7': 68,
     '8': 68, '9': 68},
    # 56
    {automata.DEFAULT: 56, '\n': 34,
     '\r': 34, "'": 15, '\\': 36},
    # 57
    {automata.DEFAULT: 57, '\n': 34,
     '\r': 34, '"': 15, '\\': 39},
    # 58
    {'0': 69, '1': 69, '2': 69, '3': 69,
     '4': 69, '5': 69, '6': 69, '7': 69,
     '8': 69, '9': 69, 'A': 69, 'B': 69,
     'C': 69, 'D': 69, 'E': 69, 'F': 69,
     'a': 69, 'b': 69, 'c': 69, 'd': 69,
     'e': 69, 'f': 69},
    # 59
    {'0': 59, '1': 59, '2': 59, '3': 59,
     '4': 59, '5': 59, '6': 59, '7': 59,
     '8': 59, '9': 59, 'A': 59, 'B': 59,
     'C': 59, 'D': 59, 'E': 59, 'F': 59,
     '_': 70, 'a': 59, 'b': 59, 'c': 59,
     'd': 59, 'e': 59, 'f': 59},
    # 60
    {'0': 71, '1': 71, '2': 71, '3': 71,
     '4': 71, '5': 71, '6': 71, '7': 71},
    # 61
    {'0': 61, '1': 61, '2': 61, '3': 61,
     '4': 61, '5': 61, '6': 61, '7': 61,
     '_': 72},
    # 62
    {'0': 73, '1': 73},
    # 63
    {'0': 63, '1': 63, '_': 74},
    # 64
    {'0': 75, '1': 75, '2': 75, '3': 75,
     '4': 75, '5': 75, '6': 75, '7': 75,
     '8': 75, '9': 75},
    # 65
    {'0': 66, '1': 66, '2': 66, '3': 66,
     '4': 66, '5': 66, '6': 66, '7': 66,
     '8': 66, '9': 66},
    # 66
    {'0': 66, '1': 66, '2': 66, '3': 66,
     '4': 66, '5': 66, '6': 66, '7': 66,
     '8': 66, '9': 66, 'J': 15, '_': 76,
     'j': 15},
    # 67
    {'0': 77, '1': 77, '2': 77, '3': 77,
     '4': 77, '5': 77, '6': 77, '7': 77,
     '8': 77, '9': 77},
    # 68
    {'0': 68, '1': 68, '2': 68, '3': 68,
     '4': 68, '5': 68, '6': 68, '7': 68,
     '8': 68, '9': 68, 'E': 51, 'J': 15,
     '_': 55, 'e': 51, 'j': 15},
    # 69
    {'0': 69, '1': 69, '2': 69, '3': 69,
     '4': 69, '5': 69, '6': 69, '7': 69,
     '8': 69, '9': 69, 'A': 69, 'B': 69,
     'C': 69, 'D': 69, 'E': 69, 'F': 69,
     '_': 58, 'a': 69, 'b': 69, 'c': 69,
     'd': 69, 'e': 69, 'f': 69},
    # 70
    {'0': 78, '1': 78, '2': 78, '3': 78,
     '4': 78, '5': 78, '6': 78, '7': 78,
     '8': 78, '9': 78, 'A': 78, 'B': 78,
     'C': 78, 'D': 78, 'E': 78, 'F': 78,
     'a': 78, 'b': 78, 'c': 78, 'd': 78,
     'e': 78, 'f': 78},
    # 71
    {'0': 71, '1': 71, '2': 71, '3': 71,
     '4': 71, '5': 71, '6': 71, '7': 71,
     '_': 60},
    # 72
    {'0': 79, '1': 79, '2': 79, '3': 79,
     '4': 79, '5': 79, '6': 79, '7': 79},
    # 73
    {'0': 73, '1': 73, '_': 62},
    # 74
    {'0': 80, '1': 80},
    # 75
    {'0': 75, '1': 75, '2': 75, '3': 75,
     '4': 75, '5': 75, '6': 75, '7': 75,
     '8': 75, '9': 75, 'E': 51, 'J': 15,
     '_': 64, 'e': 51, 'j': 15},
    # 76
    {'0': 81, '1': 81, '2': 81, '3': 81,
     '4': 81, '5': 81, '6': 81, '7': 81,
     '8': 81, '9': 81},
    # 77
    {'0': 77, '1': 77, '2': 77, '3': 77,
     '4': 77, '5': 77, '6': 77, '7': 77,
     '8': 77, '9': 77, 'J': 15, '_': 67,
     'j': 15},
    # 78
    {'0': 78, '1': 78, '2': 78, '3': 78,
     '4': 78, '5': 78, '6': 78, '7': 78,
     '8': 78, '9': 78, 'A': 78, 'B': 78,
     'C': 78, 'D': 78, 'E': 78, 'F': 78,
     '_': 70, 'a': 78, 'b': 78, 'c': 78,
     'd': 78, 'e': 78, 'f': 78},
    # 79
    {'0': 79, '1': 79, '2': 79, '3': 79,
     '4': 79, '5': 79, '6': 79, '7': 79,
     '_': 72},
    # 80
    {'0': 80, '1': 80, '_': 74},
    # 81
    {'0': 81, '1': 81, '2': 81, '3': 81,
     '4': 81, '5': 81, '6': 81, '7': 81,
     '8': 81, '9': 81, 'J': 15, '_': 76,
     'j': 15},
    ]
pseudoDFA = automata.DFA(states, accepts)

accepts = [False, False, False, False, False, True]
states = [
    # 0
    {automata.DEFAULT: 0, '"': 1, '\\': 2},
    # 1
    {automata.DEFAULT: 4, '"': 3, '\\': 2},
    # 2
    {automata.DEFAULT: 4},
    # 3
    {automata.DEFAULT: 4, '"': 5, '\\': 2},
    # 4
    {automata.DEFAULT: 4, '"': 1, '\\': 2},
    # 5
    {automata.DEFAULT: 4, '"': 5, '\\': 2},
    ]
double3DFA = automata.NonGreedyDFA(states, accepts)

accepts = [False, False, False, False, False, True]
states = [
    # 0
    {automata.DEFAULT: 0, "'": 1, '\\': 2},
    # 1
    {automata.DEFAULT: 4, "'": 3, '\\': 2},
    # 2
    {automata.DEFAULT: 4},
    # 3
    {automata.DEFAULT: 4, "'": 5, '\\': 2},
    # 4
    {automata.DEFAULT: 4, "'": 1, '\\': 2},
    # 5
    {automata.DEFAULT: 4, "'": 5, '\\': 2},
    ]
single3DFA = automata.NonGreedyDFA(states, accepts)

accepts = [False, True, False, False]
states = [
    # 0
    {automata.DEFAULT: 0, "'": 1, '\\': 2},
    # 1
    {},
    # 2
    {automata.DEFAULT: 3},
    # 3
    {automata.DEFAULT: 3, "'": 1, '\\': 2},
    ]
singleDFA = automata.DFA(states, accepts)

accepts = [False, True, False, False]
states = [
    # 0
    {automata.DEFAULT: 0, '"': 1, '\\': 2},
    # 1
    {},
    # 2
    {automata.DEFAULT: 3},
    # 3
    {automata.DEFAULT: 3, '"': 1, '\\': 2},
    ]
doubleDFA = automata.DFA(states, accepts)

