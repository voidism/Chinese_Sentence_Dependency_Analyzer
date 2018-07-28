# -*- coding: UTF-8 -*-
from anytree import Node, RenderTree
from anytree.exporter import DotExporter

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alpha = "qazwsxedcrfvtgbyhnujmikoolpQAZWSXEDCRFVTGBYHNUJMIKOLP"
mark = "()|:"
blank = " "
dot = "‧"
bottomline = '_'
# inputdata="VP(evaluation:Dbb:再|Head:VE2:想到|goal:S(agent:NP(Head:Nab:蝴蝶)|epistemics:Dbaa:會|Head:VC31:生|theme:NP(property:NP‧的(head:NP(quantifier:NP(Head:Neqa:滿)|Head:Nab:屋)|Head:DE:的)|Head:Nab:毛蟲)))"


def parsing(inputdata):
    stack = []
    rootdata = ''
    i = 0
    while(inputdata[i] != '('):
        rootdata += inputdata[i]
        i += 1
    root = Node(rootdata)
    tempwords = []
    tempwords.append(root)
    while i < len(inputdata)-1:
        if(inputdata[i] == '('or inputdata[i] == '|'):
            if(inputdata[i] == '('):
                stack.append(len(tempwords)-1)
            i += 1
            tempword = ''
            while(inputdata[i] in alpha or inputdata[i] == ':' and inputdata[i] != ')' or inputdata[i] in number or inputdata[i] in bottomline):
                if(inputdata[i] != ':'):
                    tempword += inputdata[i]
                else:
                    tempword += ' '
                if(inputdata[i+1] in dot):
                    i += 1
                    while(inputdata[i] not in alpha and inputdata[i] not in mark):
                        tempword += inputdata[i]
                        i += 1
                    i -= 1
                i += 1
            tempword2 = ''
            for k in range(len(tempword)):
                if(tempword[k] in blank and k == len(tempword)-1):
                    pass
                else:
                    tempword2 += tempword[k]
            tempwords.append(
                Node(tempword2, parent=tempwords[stack[len(stack)-1]]))
            tempword = ''
            flag = False
            while(inputdata[i] not in alpha and inputdata[i] not in mark and i < len(inputdata)-1):
                tempword += inputdata[i]
                flag = True
                i += 1
            if(flag == True):
                tempwords.append(
                    Node(tempword, parent=tempwords[len(tempwords)-1]))
            i -= 1

        elif(inputdata[i] == ')'):
            if not stack:
                return -1
            stack.pop()
        i += 1
    stack.pop()
    if stack:
        return -1
    return root


def main(sent="VP(evaluation:Dbb:再|Head:VH_11:想到|goal:S(agent:NP(Head:Nab:蝴蝶)|epistemics:Dbaa:會|Head:VC31:生|theme:NP(property:NP‧的(head:NP(quantifier:NP(Head:Neqa:滿)|Head:Nab:屋)|Head:DE:的)|Head:Nab:毛蟲)))"):
    tree = parsing(sent)
    if(tree == -1):
        print("Wrong input")
    else:
        for pre, fill, node in RenderTree(tree):
            print("%s%s" % (pre, node.name))
    return tree


if __name__ == '__main__':
    main("VP(evaluation:Dbb:再|Head:VH_11:想到|goal:S(agent:NP(Head:Nab:蝴蝶)|epistemics:Dbaa:會|Head:VC31:生|theme:NP(property:NP‧的(head:NP(quantifier:NP(Head:Neqa:滿)|Head:Nab:屋)|Head:DE:的)|Head:Nab:毛蟲)))")
