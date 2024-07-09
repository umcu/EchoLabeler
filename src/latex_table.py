import os
import re
import sys
import argparse
from typing import Literal 
from collections import defaultdict
import numpy as np

standard_cleaning = lambda x: x.strip("(").strip(")")

def _parse_value_line(Line: str=None,  GroupSize: int=None, TextColNum: int=0) -> str:
    Line = Line.replace("\\\\", "")
    Elements = Line.split("&")
    ColName = Elements[0]
    ValDict = defaultdict(dict)
    MaxVal1, MaxVal2 = np.zeros(GroupSize), np.zeros(GroupSize)
    Group = 0
    for i, val_str in enumerate(Elements[1:]):
        if (i%GroupSize==0) & (i>0):
            Group +=1
        val_str = val_str.replace(" ", "")
        val_str = val_str.strip()
        vals = val_str.split("(")
        v1 = float(standard_cleaning(vals[0]))
        v2 = float(standard_cleaning(vals[1]))
        ValDict[Group][i%3] = (v1,v2)        
        MaxVal1[i%3] = max(MaxVal1[i%3], v1)
        MaxVal2[i%3] = max(MaxVal2[i%3], v2)

    OutStr = Elements[0] + "&"
    for group, vs in ValDict.items():
        for j, (v1,v2) in vs.items():
            if v1==MaxVal1[j]:
                v1Str = "\\textbf{"+str(v1)+"}"
            else:
                v1Str = str(v1)

            if v2==MaxVal2[j]:
                v2Str = "\\textbf{"+str(v2)+"}"
            else:
                v2Str = str(v2)      
            
            vStr = v1Str + "(" + v2Str + ")"
            
            OutStr += vStr + "&"
    OutStr = OutStr.strip("&") + "\\\\"
    
    return OutStr       
    
    
def parse_table(Path: str=None, OutPath: str=None, parse_type: Literal['boldface']=None, GroupSize: int=None, TextColNum: int=0, FirstTabStr: str=None):
    HeaderStr = ""
    TableStr = ""
    FooterStr = ""

    InHeader = True
    InTable = False
    InFooter = False
    with open(Path, mode='r') as FileReader:
        for line in FileReader:
            if ((FirstTabStr in line) & (InFooter==False)) | (InTable==True):
                InHeader = False
                InTable = True
                InFooter = False
            if ("\\end" in line) | (InFooter==True):
                InHeader = False
                InTable  = False
                InFooter = True
            
            if InHeader:
                HeaderStr += line
            
            if InTable:
                TableStr += _parse_value_line(line, GroupSize=GroupSize, TextColNum=0) + "\r"
            
            if InFooter:
                FooterStr += line
    
    # write out edited file
    with open(OutPath, mode='w') as FileWriter:        
        FileWriter.write(HeaderStr+""+TableStr+""+FooterStr)
    
    return True


if __name__ ==  "__main__" :
    # add argument parser
    ArgParse = argparse.ArgumentParser(description="Basic arg parser")
    ArgParse.add_argument("-i", "--input", type=str, required=True, dest="text_file_to_parse")
    ArgParse.add_argument("-o", "--output", type=str, required=True, dest="output_file")
    ArgParse.add_argument("-t", "--type", type=str, choices=["boldface"], dest="parse_type", required=False, default="boldface")
    
    text_file_to_parse = ArgParse.parse_args().text_file_to_parse
    output_file = ArgParse.parse_args().output_file
    parse_type = ArgParse.parse_args().parse_type
    
    FirstTabStr = 'diastolic dysfunction'
    GroupSize = 3
    parse_table(Path=text_file_to_parse, OutPath=output_file, parse_type=parse_type, FirstTabStr=FirstTabStr, GroupSize=GroupSize)