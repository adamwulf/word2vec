//
//  Word2Vec.m
//  Word2Vec
//
//  Created by Adam Wulf on 1/6/18.
//

#import "Word2Vec.h"



#if defined __cplusplus
extern "C" {
#endif

#include "trainingThread.c"
#include "vocab.c"
#include "ngram_tools.c"
#include "word2vec-inc.h"
#include "word2vec.c"
    
#if defined __cplusplus
};
#endif


@implementation Word2Vec

-(instancetype _Nonnull ) initWithTrainFile:(NSURL*_Nonnull)trainFile andOutputFile:(NSURL*_Nonnull)outputFile{
    if(self = [super init]){
        _trainFile = trainFile;
        _outputFile = outputFile;
    }
    return self;
}

-(BOOL) train{
    if(![self trainFile] || ![self outputFile]){
        @throw [NSException exceptionWithName:@"Word2VecException" reason:@"Both trainFile and outputFile are required." userInfo:nil];
    }
    
    return [Word2Vec trainwithCorpusFile:[self trainFile]
                              outputFile:[self outputFile]
                           saveVocabFile:[self saveVocabFile]
                           readVocabFile:[self readVocabFile]
                          wordVectorSize:[self wordVectorSize]
                                   debug:[self debug]
                            saveToBinary:[self saveToBinary]
                    continuousBagOfWords:[self continuousBagOfWords]
                    startingLearningRate:[self startingLearningRate]
                            windowLength:[self windowLength]
                wordsOccurrenceThreshold:[self wordsOccurrenceThreshold]
                     hierarchicalSoftmax:[self hierarchicalSoftmax]
                        negativeExamples:[self negativeExamples]
                                 threads:[self threads]
                      trainingIterations:[self trainingIterations]
                                minCount:[self minCount]
                           classesNumber:[self classesNumber]];
}

#pragma mark - main

+ (BOOL)trainwithCorpusFile:(NSURL * _Nonnull) trainFile
                 outputFile:(NSURL * _Nonnull) outputFile
              saveVocabFile:(NSURL * _Nullable) saveVocabFile
              readVocabFile:(NSURL * _Nullable) readVocabFile
             wordVectorSize:(NSNumber * _Nullable) wordVectorSize
                      debug:(NSNumber * _Nullable) debug
               saveToBinary:(NSNumber * _Nullable) saveToBinary
       continuousBagOfWords:(NSNumber * _Nullable) continuousBagOfWords
       startingLearningRate:(NSNumber * _Nullable) startingLearningRate
               windowLength:(NSNumber * _Nullable) windowLength
   wordsOccurrenceThreshold:(NSNumber * _Nullable) wordsOccurrenceThreshold
        hierarchicalSoftmax:(NSNumber * _Nullable) hierarchicalSoftmax
           negativeExamples:(NSNumber * _Nullable) negativeExamples
                    threads:(NSNumber * _Nullable) threads
         trainingIterations:(NSNumber * _Nullable) trainingIterations
                   minCount:(NSNumber * _Nullable) minCount
              classesNumber:(NSNumber * _Nullable) classesNumber {
    NSFileManager *manager = [NSFileManager defaultManager];
    
    char const *trainFilePath = [manager fileSystemRepresentationWithPath:trainFile.path];
    char const *outputFilePath = [manager fileSystemRepresentationWithPath:outputFile.path];

    // check for MAX_STRING
    strcpy(train_file, trainFilePath);
    strcpy(output_file, outputFilePath);
    
    if (saveVocabFile) {
        char const *saveVocabFilePath = [manager fileSystemRepresentationWithPath:saveVocabFile.path];
        
        strcpy(save_vocab_file, saveVocabFilePath);
    }
    
    if (readVocabFile) {
        char const *readVocabFilePath = [manager fileSystemRepresentationWithPath:readVocabFile.path];
        
        strcpy(read_vocab_file, readVocabFilePath);
    }
    if (wordVectorSize) layer1_size = wordVectorSize.longLongValue;
    if (debug) debug_mode = debug.intValue;
    if (saveToBinary) binary = saveToBinary.intValue;
    
    if (continuousBagOfWords) cbow = continuousBagOfWords.intValue;
    if (continuousBagOfWords) { alpha = 0.05; }
    if (startingLearningRate) alpha = startingLearningRate.floatValue;
    
    if (windowLength) window = windowLength.intValue;
    if (wordsOccurrenceThreshold) sample = wordsOccurrenceThreshold.floatValue;
    if (hierarchicalSoftmax) hs = hierarchicalSoftmax.intValue;
    if (negativeExamples) negative = negativeExamples.intValue;
    if (threads) num_threads = threads.intValue;
//    if (trainingIterations) iter = trainingIterations.longLongValue; // interations is always 10
    if (minCount) min_count = minCount.intValue;
    if (classesNumber) classes = classesNumber.longLongValue;
    
    
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    /**
     Fixed starting Parameters:
     **/
    int vocab_hash_size =  30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
    int vocab_max_size = 1000;
    
    //1: init vocabulary
    vocabulary* vocab = InitVocabulary(vocab_hash_size,vocab_max_size);
    
    //2: load vocab
    if (read_vocab_file[0] != 0)
        file_size = ReadVocab(vocab,read_vocab_file,train_file,min_count);
    else
        file_size = LearnVocabFromTrainFile(vocab,train_file,min_count);
    
    if (save_vocab_file[0] != 0)
        SaveVocab(vocab,save_vocab_file);
    
    if (output_file[0] == 0) //nowhere to output => quit
        return NO;
    
    //3: train_model
    TrainModel(vocab);
    
    free(expTable);
    DestroyNet();
    DestroyVocab(vocab);
    
    return YES;
}

@end
