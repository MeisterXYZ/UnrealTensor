import os

lineNo=1
outputLine = ''

with open('ORIGINstupidTestClick.txt') as inputFile:
    with open ('stupidTestClickTransformed.txt' , 'w') as f:
        for line in inputFile:
            #print(line[:-2])
            outputLine = outputLine+line[:-1]
            lineNo = lineNo + 1
            if lineNo % 100 == 0:
                print(outputLine)
                #write outputLine with label as last element
                f.write(outputLine+'1\n')
                outputLine = '' 