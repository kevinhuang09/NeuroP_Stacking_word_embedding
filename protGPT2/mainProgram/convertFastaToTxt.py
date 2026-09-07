from Bio import SeqIO

class convertFastaToTxt:
    def __init__(self, origPepFastaPath, outputPepTxtPath):
        self.origPepFastaPath = origPepFastaPath
        self.outputPepTxtPath = outputPepTxtPath

    def fastaToList(self, file_path):
        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))
        return sequences

    def main(self):
        origPeptideFile = self.fastaToList(self.origPepFastaPath)
        genPepFile = open(self.outputPepTxtPath, "w")

        seqLengthList = []
        for sequence in origPeptideFile:
            sequence = sequence.rstrip("\n")
            aaSet = set("ACDEFGHIKLMNPQRSTVWY")
            if all(aa.upper() in aaSet for aa in sequence):
                genPepFile.write("<|endoftext|>" + "\n" + str(sequence) + "\n")
                seqLengthList.append(len(sequence))
            else:
                print("Sequence contains non-amino acid:", sequence)

        genPepFile.close()

        if seqLengthList:
            maxLength = max(seqLengthList)
            minLength = min(seqLengthList)
            print("Maximum sequence length:", maxLength)
            print("Minimum sequence length:", minLength)
        else:
            print("No valid sequences found.")

convertFastaToTxt = convertFastaToTxt(origPepFastaPath = "../data/IL13/IL13pos_orig.fasta",
                                      outputPepTxtPath = "../data/IL13/IL13pos_all.txt"
                                      )
convertFastaToTxt.main()