# NLP Assignment 2
# Jing Jiang
# Reference to Wikipedia page of CYK algorithm: https://en.wikipedia.org/wiki/CYK_algorithm

import sys
import string

class Node:
    def __init__(self, key, child1, child2):
        self.key = key
        self.child1 = child1
        self.child2 = child2

def read_cnf(cnf):
    infile = open(str(cnf[1]), 'r')
    cnf1 = {}
    cnf2 = {}
    for line in infile:
        words = line.strip().split()
        if len(words) == 3:
            if words[2] not in cnf1.keys():
                cnf1[words[2]] = set()
            cnf1[words[2]].add(words[0])
        elif len(words) == 4:
            if (words[2], words[3]) not in cnf2.keys():
                cnf2[(words[2], words[3])] = set()
            cnf2[(words[2], words[3])].add(words[0])
    infile.close()
    return cnf1, cnf2

def cyk(sentence, cnf1, cnf2):
    num = len(sentence)
    result = []
    parse = {}
    for i in range(num+1):
        parse[i] = {}
        for j in range(num+1):
            parse[i][j] = []

    for j in range(1, num+1):
        if sentence[j-1] in cnf1.keys():
            for word in cnf1[sentence[j-1]]:
                parse[j-1][j].append(Node(word, Node(sentence[j-1], None, None), None))
        for i in range(j-2, -1, -1):
            for k in range(i+1, j):
                for B in parse[i][k]:
                    for C in parse[k][j]:
                        if (B.key, C.key) in list(cnf2.keys()):
                            for word in cnf2[(B.key, C.key)]:
                                parse[i][j].append(Node(word, B, C))
    for node in parse[0][num]:
        if node.key == 'S':
            result.append(node)

    return result

def runcyk(cnf1, cnf2):
    while True:
        sentence = input('Enter a sentence to Parse: ')
        if sentence == 'quit':
            break

        result = cyk(sentence.split(), cnf1, cnf2)

        if len(result) == 0:
            sys.stderr.write("No valid parse found.\n")
        else:
            pnum = 1
            for node in result:
                out_str = ""
                out_str = print_parse(node, out_str)
                sys.stdout.write('Parse No.' + str(pnum) + ': ' + out_str + '\n')
                pnum += 1

def print_parse(node, result_str):
    if node.child1:
        if result_str.endswith(']'):
            result_str = result_str + ' [' + node.key + ' '
        else:
            result_str = result_str + '[' + node.key + ' '
        result_str = print_parse(node.child1, result_str)
    else:
        result_str = result_str + node.key + ']'
    if node.child2:
        result_str = print_parse(node.child2, result_str)
        result_str = result_str + ']'
    
    return result_str

if __name__ == "__main__":
    cnf1, cnf2 = read_cnf(sys.argv)
    runcyk(cnf1, cnf2)
