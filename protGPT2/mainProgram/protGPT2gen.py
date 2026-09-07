from transformers import pipeline

class Protgpt2:
    def __init__(self, origFile, outputPath, foldNum, model, maxlenth, minlenth, datasetName, outputSuffix):
        self.origFile = origFile
        self.outputPath = outputPath
        self.foldNum = foldNum
        self.model = model
        self.maxlenth = maxlenth
        self.minlenth = minlenth
        self.datasetName = datasetName
        self.outputSuffix = outputSuffix

    def listToFasta(self, lst, output_file, peptideNameNumStart):
        with open(output_file, 'w') as file:
            if peptideNameNumStart != None:
                for sequence in lst:
                    file.write(f'>GPT{self.outputSuffix}{peptideNameNumStart+1}\n{sequence}\n')
                    peptideNameNumStart += 1
            else:
                num = 1
                for sequence in lst:
                    file.write(f'>GPT{self.outputSuffix}{num}\n{sequence}\n')
                    num += 1

    def filterNonAminoAcids_or_Existed_or_WrongLenth(self, sequence, result, accumulatedList):
        amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        #改一維
        for aa in sequence:
            if aa.upper() not in amino_acids or sequence in result or sequence in accumulatedList or len(sequence)<self.minlenth or len(sequence)>self.maxlenth:
                return False
        return sequence

    def gen(self, start,protgpt2):
        seq = protgpt2(f"{start}", max_new_tokens=9, min_new_tokens=3, do_sample=True, top_k=950, repetition_penalty=1.2,
                       num_return_sequences=1, eos_token_id=0)
        return seq

    def readFastaToList(self, fileInput, origPepList):
        count = 1
        for strLine in fileInput:
            strLine = strLine.rstrip("\n")
            if count % 2 == 0:
                origPepList.append(strLine)
            count += 1
        return origPepList

    def main(self):
        fileInput = open(self.origFile, "r")
        origPepList = []
        accumulatedPepList = []
        origPepList = self.readFastaToList(fileInput, origPepList)
        peptideNameNumEnd = len(origPepList)
        peptideNameNumStart = peptideNameNumEnd
        accumulatedPepList.append(origPepList)
        for i in range(1, self.foldNum + 1):
            foldPepList = []
            #test = 0 #維修用
            for pepStr in origPepList:
                isValid = False
                startAA = pepStr[0]
                protgpt2 = pipeline('text-generation', model=self.model)
                while isValid == False:
                    genPepDict = self.gen(startAA,protgpt2)[0]
                    genPepStr = genPepDict["generated_text"]
                    genPepStr = genPepStr.replace("\n", "")
                    isValid = self.filterNonAminoAcids_or_Existed_or_WrongLenth(genPepStr, foldPepList, accumulatedPepList)
                foldPepList.append(genPepStr)
                peptideNameNumEnd += 1
                print(genPepStr)
                #test+=1 #維修用
                #if test == 3:
                    #break
            self.listToFasta(foldPepList, f"{self.outputPath}{self.datasetName}x{i}{self.outputSuffix}_gpt.fasta", peptideNameNumStart)
            peptideNameNumStart = peptideNameNumEnd
            accumulatedPepList.append(foldPepList)
            accumulatedPepFlattenList = []
            for peptide in accumulatedPepList:
                accumulatedPepFlattenList.extend(peptide)
            self.listToFasta(accumulatedPepFlattenList, f"{self.outputPath}{self.datasetName}x{i}{self.outputSuffix}_accumulated_gpt.fasta",None)

protgpt2Obj = Protgpt2(origFile="../data/IL13/IL13pos_all.txt",
                       outputPath="../data/Il13/",
                       model="../model/IL13",
                       foldNum= 9,
                       maxlenth= 35,
                       minlenth = 8,
                       datasetName="IL13",
                       outputSuffix="pos") #不需要就留空
protgpt2Obj.main()
