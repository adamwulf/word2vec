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

static NSString * const MemoryAllocationError = @"MemoryAllocationError";

const long long maxStringLength = 2000;         // max length of strings
long long numberToShow = 40;                  // number of closest words that will be shown
const long long entryMaxLength = 50;              // max length of vocabulary entries

@interface Word2Vec (){
    char file_name[maxStringLength];
    char *vocab;
    float *M;
    
}

@property (nonatomic) FILE *modelFile;
@property (nonatomic, strong) NSNumber *wordsTotalNum; // long long
@property (nonatomic, strong) NSNumber *size; // long long

@end

@implementation Word2Vec

-(instancetype _Nonnull ) initWithTrainFile:(NSURL*_Nonnull)trainFile andOutputFile:(NSURL*_Nonnull)outputFile{
    if(self = [super init]){
        _trainFile = trainFile;
        _outputFile = outputFile;
    }
    return self;
}

-(void) setOutputFile:(NSURL *)outputFile{
    if(M){
        @throw [NSException exceptionWithName:@"Word2VecException" reason:@"Cannot change word vector model file once already loaded." userInfo:nil];
    }
    
    _outputFile = outputFile;
}

- (void)loadBinaryVectorFile:(NSURL * _Nonnull) fileURL
                       error:(NSError ** _Nullable) error {
    
    NSFileManager *manager = [NSFileManager defaultManager];
    char const *outputFilePath = [manager fileSystemRepresentationWithPath:fileURL.path];
    
    strcpy(file_name, outputFilePath);
    self.modelFile = fopen(file_name, "rb");
    
    long long wordsTotalNum, size;
    
    fscanf(self.modelFile, "%lld", &wordsTotalNum);
    fscanf(self.modelFile, "%lld", &size);
    
    self.wordsTotalNum = @(wordsTotalNum);
    self.size = @(size);
    
    
    vocab = (char *)malloc((long long)wordsTotalNum * entryMaxLength * sizeof(char));
    
    M = (float *)malloc((long long)wordsTotalNum * (long long)size * sizeof(float));
    
    if (M == NULL) {
        NSString *message = [NSString stringWithFormat:@"Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)wordsTotalNum * size * sizeof(float) / 1048576, wordsTotalNum, size];
        
        * error = [NSError errorWithDomain:MemoryAllocationError
                                      code:1
                                  userInfo:@{NSLocalizedDescriptionKey : message}];
        
        NSLog(@"%@", message);
        return;
    }
    
    long long a, b;
    float len;
    
    for (b = 0; b < wordsTotalNum; b++) {
        a = 0;
        while (1) {
            vocab[b * entryMaxLength + a] = fgetc(self.modelFile);
            if (feof(self.modelFile) || (vocab[b * entryMaxLength + a] == ' ')) break;
            if ((a < entryMaxLength) && (vocab[b * entryMaxLength + a] != '\n')) a++;
        }
        vocab[b * entryMaxLength + a] = 0;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, self.modelFile);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(self.modelFile);
}

-(void) loadWordVectors{
    if(!M){
        NSError* error;
        [self loadBinaryVectorFile:[self outputFile] error:&error];
        
        if(error){
            @throw [NSException exceptionWithName:@"Word2VecException" reason:@"Could not load word vectors" userInfo:nil];
        }
    }
}

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)closestToWord:(NSString * _Nonnull) word
                                                    numberOfClosest:(NSNumber * _Nullable) numberOfClosest {
    [self loadWordVectors];
    
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    
    if(numberOfClosest) { numberToShow = numberOfClosest.longLongValue; }
    
    char st1[maxStringLength];
    char *bestWords[numberToShow];
    char st[100][maxStringLength];
    float dist, len, bestDistances[numberToShow], vec[self.size.longLongValue];
    long long a, b, c, d, cn, bi[100];
    
    for (a = 0; a < numberToShow; a++) {
        bestWords[a] = (char *)malloc(maxStringLength * sizeof(char));
    }
    
    for (a = 0; a < numberToShow; a++) bestDistances[a] = 0;
    for (a = 0; a < numberToShow; a++) bestWords[a][0] = 0;
    a = 0;
    while (1) {
        st1[a] = [word cStringUsingEncoding:NSUTF8StringEncoding][a];
        if ((st1[a] == '\0') || (st1[a] == '\n') || (a >= maxStringLength - 1)) {
            st1[a] = 0;
            break;
        }
        a++;
    }
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
        st[cn][b] = st1[c];
        b++;
        c++;
        st[cn][b] = 0;
        if (st1[c] == 0) break;
        if (st1[c] == ' ') {
            cn++;
            b = 0;
            c++;
        }
    }
    cn++;
    
    long long wordsTotalNum = self.wordsTotalNum.longLongValue;
    
    for (a = 0; a < cn; a++) {
        for (b = 0; b < wordsTotalNum; b++) {
            if (!strcmp(&vocab[b * entryMaxLength], st[a])) {
                break;
            }
        }
        if (b == wordsTotalNum) b = -1;
        bi[a] = b;
        
        if (b == -1) {
            NSLog(@"Out of dictionary word!\n");
            break;
        }
    }
    
    long long size = self.size.longLongValue;
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
        if (bi[b] == -1) continue;
        for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < numberToShow; a++) bestDistances[a] = -1;
    for (a = 0; a < numberToShow; a++) bestWords[a][0] = 0;
    for (c = 0; c < wordsTotalNum; c++) {
        a = 0;
        for (b = 0; b < cn; b++) {
            if (bi[b] == c) {
                a = 1;
            }
        }
        if (a == 1) continue;
        dist = 0;
        for (a = 0; a < size; a++) {
            dist += vec[a] * M[a + c * size];
        }
        for (a = 0; a < numberToShow; a++) {
            if (dist > bestDistances[a]) {
                for (d = numberToShow - 1; d > a; d--) {
                    bestDistances[d] = bestDistances[d - 1];
                    strcpy(bestWords[d], bestWords[d - 1]);
                }
                bestDistances[a] = dist;
                strcpy(bestWords[a], &vocab[c * entryMaxLength]);
                break;
            }
        }
    }
    
    for (a = 0; a < numberToShow; a++) {
        result[[NSString stringWithCString:bestWords[a] encoding:NSUTF8StringEncoding]] = @(bestDistances[a]);
    }
    
    free(*bestWords);
    return result;
}

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)analogyToPhrase:(NSString * _Nonnull) phrase
                                                      numberOfClosest:(NSNumber * _Nullable) numberOfClosest {
    [self loadWordVectors];

    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if(numberOfClosest) { numberToShow = numberOfClosest.longLongValue; }
    char st1[maxStringLength];
    char *bestWords[numberToShow];
    char st[100][maxStringLength];
    float dist, len, bestDistances[numberToShow], vec[self.size.longLongValue];
    long long a, b, c, d, cn, bi[100];
    
    for (a = 0; a < numberToShow; a++) {
        bestWords[a] = (char *)malloc(maxStringLength * sizeof(char));
    }
    for (a = 0; a < numberToShow; a++) bestDistances[a] = 0;
    for (a = 0; a < numberToShow; a++) bestWords[a][0] = 0;
    a = 0;
    while (1) {
        st1[a] = [phrase cStringUsingEncoding:NSUTF8StringEncoding][a];
        if ((st1[a] == '\0') || (st1[a] == '\n') || (a >= maxStringLength - 1)) {
            st1[a] = 0;
            break;
        }
        a++;
    }
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
        st[cn][b] = st1[c];
        b++;
        c++;
        st[cn][b] = 0;
        if (st1[c] == 0) break;
        if (st1[c] == ' ') {
            cn++;
            b = 0;
            c++;
        }
    }
    cn++;
    
    long long wordsTotalNum = self.wordsTotalNum.longLongValue;
    
    for (a = 0; a < cn; a++) {
        for (b = 0; b < wordsTotalNum; b++) if (!strcmp(&vocab[b * entryMaxLength], st[a])) break;
        if (b == wordsTotalNum) b = 0;
        bi[a] = b;
        if (b == 0) {
            NSLog(@"Out of dictionary word!\n");
            break;
        }
    }
    
    long long size = self.size.longLongValue;
    
    NSInteger wordNumber = [phrase componentsSeparatedByString:@" "].count;
    if (wordNumber == 2) {
        for (a = 0; a < size; a++) vec[a] = M[a + bi[0] * size] - M[a + bi[1] * size];
    } else {
        for (a = 0; a < size; a++) vec[a] = M[a + bi[1] * size] - M[a + bi[0] * size] + M[a + bi[2] * size];
    }
    
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < numberToShow; a++) bestDistances[a] = 0;
    for (a = 0; a < numberToShow; a++) bestWords[a][0] = 0;
    for (c = 0; c < wordsTotalNum; c++) {
        if (c == bi[0]) continue;
        if (c == bi[1]) continue;
        if (c == bi[2]) continue;
        a = 0;
        for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
        if (a == 1) continue;
        dist = 0;
        for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
        for (a = 0; a < numberToShow; a++) {
            if (dist > bestDistances[a]) {
                for (d = numberToShow - 1; d > a; d--) {
                    bestDistances[d] = bestDistances[d - 1];
                    strcpy(bestWords[d], bestWords[d - 1]);
                }
                bestDistances[a] = dist;
                strcpy(bestWords[a], &vocab[c * entryMaxLength]);
                break;
            }
        }
    }
    
    for (a = 0; a < numberToShow; a++) {
        result[[NSString stringWithCString:bestWords[a] encoding:NSUTF8StringEncoding]] = @(bestDistances[a]);
    }
    
    free(*bestWords);
    return result;
}

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)wordSimilarity:(NSString * _Nonnull) phrase {
    [self loadWordVectors];
    
    numberToShow = 1;
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    char full_input_string[maxStringLength];
    char input_words[100][maxStringLength];
    float dist, len, vec[self.size.longLongValue];
    long long i, b, c, d, word_count, input_word_index[100];
    
    i = 0;
    while (1) {
        full_input_string[i] = [phrase cStringUsingEncoding:NSUTF8StringEncoding][i];
        if ((full_input_string[i] == '\0') || (full_input_string[i] == '\n') || (i >= maxStringLength - 1)) {
            full_input_string[i] = 0;
            break;
        }
        i++;
    }
    word_count = 0;
    b = 0;
    c = 0;
    while (1) {
        input_words[word_count][b] = full_input_string[c];
        b++;
        c++;
        input_words[word_count][b] = 0;
        if (full_input_string[c] == 0) break;
        if (full_input_string[c] == ' ') {
            word_count++;
            b = 0;
            c++;
        }
    }
    word_count++;
    
    // initialize our distances to 0 for all our input words
    float bestDistances[word_count];
    for (i = 0; i < word_count; i++) bestDistances[i] = 0;

    // iterate over all our vocabulary to find our input word's indexes in our vocab
    long long countInVocab = self.wordsTotalNum.longLongValue;
    
    for (i = 0; i < word_count; i++) {
        for (b = 0; b < countInVocab; b++){
            if (!strcmp(&vocab[b * entryMaxLength], input_words[i])){
                break;
            }
        }
        if (b == countInVocab) b = 0;
        input_word_index[i] = b;
        if (b == 0) {
            NSLog(@"Out of dictionary word!\n");
            break;
        }
    }
    
    long long size = self.size.longLongValue;
    
    // initialize to all zeroes
    for (i = 0; i < size; i++) vec[i] = 0;
    
    // next, add the vectors of all the input words
    for (b = 0; b < word_count; b++) {
        for (i = 0; i < size; i++){
            vec[i] += M[i + input_word_index[b] * size];
        }
    }
    
    // next, normalize it so its length is 1
    len = 0;
    for (i = 0; i < size; i++) len += vec[i] * vec[i];
    len = sqrt(len);
    for (i = 0; i < size; i++) vec[i] /= len;
    
    // now, iterate over all input words, and add their distances to our output
    for (i = 0; i < word_count; i++) bestDistances[i] = 0;
    for (c = 0; c < word_count; c++) {
        dist = 0;
        for (i = 0; i < size; i++) dist += vec[i] * M[i + input_word_index[c] * size];
        bestDistances[c] = dist;
    }
    
    for (i = 0; i < word_count; i++) {
        result[[NSString stringWithCString:input_words[i] encoding:NSUTF8StringEncoding]] = @(bestDistances[i]);
    }
    
    return result;
}

- (void)dealloc
{
    free(vocab);
    free(M);
}

-(BOOL) train{
    if(![self trainFile] || ![self outputFile]){
        @throw [NSException exceptionWithName:@"Word2VecException" reason:@"Both trainFile and outputFile are required." userInfo:nil];
    } else if(![[NSFileManager defaultManager] fileExistsAtPath:[[self trainFile] path]]){
        @throw [NSException exceptionWithName:@"Word2VecException" reason:@"trainFile does not exist." userInfo:nil];
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
    
    if(strlen(trainFilePath) > MAX_STRING){
        printf("ERROR: trainFilePath is longer than %d!\n", MAX_STRING);
        exit(1);
    }

    if(strlen(outputFilePath) > MAX_STRING){
        printf("ERROR: outputFilePath is longer than %d!\n", MAX_STRING);
        exit(1);
    }
    
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
