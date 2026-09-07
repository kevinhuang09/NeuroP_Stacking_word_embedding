import re
import random
import textwrap

class RandomPeptide:
    def __init__(self, sequences, lengthStart, lengthEnd=0):
        if lengthEnd == 0:
            dnaDict = self.PepSequence(int(sequences), int(lengthStart), int(lengthStart))
        else:
            dnaDict = self.PepSequence(int(sequences), int(lengthStart), int(lengthEnd))
        self.writeSequencesToFile("RandomPeptide.fasta")

    def genDNA(self):
        path = "./humen.fasta"
        with open(path) as f:
            records = f.read()
        records = records.split('>')[1:]
        SeqStr = []
        for fasta in records:
            array = fasta.split('\n')
            name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
            SeqStr.extend(sequence)
        return SeqStr

    def PepSequence(self, sequences, lengthStart, lengthEnd):
        self.pepDict = {}
        self.genList = self.genDNA()
        for i in range(sequences):
            pepSequence = ""
            for _ in range(random.randint(lengthStart, lengthEnd)):
                pepSequence += random.choice(self.genList)
            dnaTitle = ">Sequence " + str(i + 1) + " of " + str(sequences)
            self.pepDict[dnaTitle] = pepSequence

    def writeSequencesToFile(self, filePath):
        with open(filePath, "w") as f:
            for k, v in self.pepDict.items():
                f.write(k)
                f.write('\n')
                f.write('\n'.join(textwrap.wrap(v, 80)))
                f.write('\n')

    def checkRatio(self):
        AA = 'ARNDCQEGHILKMFPSTWYV'
        pepcount = "".join(self.pepDict.values())
        for i in AA:
            print("人造", i, pepcount.count(i)/len(pepcount))
            print("天然", i, self.genList.count(i)/len(self.genList))

a = RandomPeptide(100000, 5)
a.checkRatio()

