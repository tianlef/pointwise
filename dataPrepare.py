# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:53:51 2018

@author: tianle
"""


def convert2TSV(dataset="QA1"):
    lines=[]
    with open(dataset+'/qemax-test.txt', 'r', encoding='utf-8') as f:
        ss=f.readlines()
        for s in ss:
            item=s.strip().split(' ')
            qid=item[0]
            term=item[1]
            label=item[2]
            query1=item[4:]
            query=" ".join(query1)
            qlen=len(query1)
            #print(query)
            line= "\t".join((qid,str(qlen),query, term, label))
            lines.append(line)
            print(line)
    filename= dataset+"/original/term-test.tsv"
    with open(filename, "w",encoding='utf-8') as f:
        print(filename)
        f.write("\n".join(lines) ) 
    return filename
def qid(dataset="QA1"):
    with open(dataset+'/qeo-test.txt', 'r', encoding='utf-8') as f:
        ss=f.readlines()
        number=[]
        for s in ss:
            item=s.strip().split(" ")
            num=len(item)-1
            number.append(num)
    return number
def convert2TSVtrue(dataset="QA1"):
    lines=[]
    with open(dataset+'/qemax-test.txt', 'r', encoding='utf-8') as f:
        ss=f.readlines()
        for s in ss:
            item=s.strip().split(' ')
            qid=item[0]
            term=item[1]
            label=item[2]
            query1=item[4:]
            query=" ".join(query1)
            qlen=len(query1)
            #print(query)
            if(label=="1"):
                  line= "\t".join((qid,str(qlen),query, term, label))
                  lines.append(line)
    filename= dataset+"/original/groundTruth-test.tsv"#true
    with open(filename, "w",encoding='utf-8') as f:
        print(filename)
        f.write("\n".join(lines) ) 
    return filename
def format_file(dataset="QA1",s="term-train"):
    filename=dataset+"/original/"+s+".tsv"
    temp_file=dataset+"/qe/"+s
    print(temp_file)
    with open(filename,'r',encoding='utf-8') as f, open(temp_file,"w",encoding='utf-8') as out:
        for index, line in enumerate(f):
            qid,qlen,query,term,label=line.strip().split("\t")
            newline="%s %s qid:%s" %(label,qlen,qid)
            tokens1=query.split()+term.split()
            fill=max(0,20-len(tokens1))
            tokens1.extend(['<a>']*fill)
            newline+=" "+"_".join( tokens1)+"_"
            tokens2 = term.split()
            fill = max(0, 20 - len(tokens2))
            tokens2.extend([term] * fill)
            newline += " " + "_".join(tokens2) + "_"
            out.write(newline + "\n")



    return temp_file

if __name__ == "__main__":
	# parseTrain()
    convert2TSV(dataset="WebAP") 
    convert2TSVtrue(dataset="WebAP")
    format_file(dataset="WebAP",s="term-test")
    format_file(dataset="WebAP",s="groundTruth-test")
    number=qid(dataset="WebAP")
    print(number)
    
    
