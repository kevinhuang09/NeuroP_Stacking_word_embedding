import re
import random
import textwrap

class RandomPeptide:
    def __init__(self, sequences, minLen, maxLen=0):
        if maxLen == 0:
            self.PepSequence(int(sequences), int(minLen), int(minLen))
        else:
            self.PepSequence(int(sequences), int(minLen), int(maxLen))
        self.writeSequencesToFile("output.fasta")

    def genDNA(self):
        path = "./SwissProtHumanProtein.fasta"
        with open(path) as f:
            records = f.read()
        records = records.split('>')[1:]
        SeqStr = []
        for fasta in records:
            array = fasta.split('\n')
            name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
            if int(len(sequence)) >= self.maxLen:
                SeqStr.append(sequence)
            else:
                pass
        return SeqStr

    def PepSequence(self, sequences, minLen, maxLen):
        self.maxLen = int(maxLen)
        self.pepDict = {}
        self.genList = self.genDNA()
        for i in range(sequences):
            protSeq = random.sample(self.genList, 1)[0]
            randLen = random.randint(minLen, maxLen)
            if int(len(protSeq))-int(maxLen) > 0:
                protStart = int(len(protSeq))-int(maxLen)
                proteinStart = random.randrange(0, protStart)
            else:
                proteinStart = int(len(protSeq))-int(maxLen)


            pepSequence = protSeq[int(proteinStart):int(proteinStart+randLen)]
            dnaTitle = ">Sequence " + str(i + 1) + " of " + str(sequences)
            self.pepDict[dnaTitle] = pepSequence

    def writeSequencesToFile(self, filePath):
        with open(filePath, "w") as f:
            for k, v in self.pepDict.items():
                f.write(k)
                f.write('\n')
                f.write('\n'.join(textwrap.wrap(v, 80)))
                f.write('\n')


RandomPeptide(100000, 10, 20)


