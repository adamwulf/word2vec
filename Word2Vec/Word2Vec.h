//
//  Word2Vec.h
//  Word2Vec
//
//  Created by Adam Wulf on 1/6/18.
//

#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>

//! Project version number for Word2Vec.
FOUNDATION_EXPORT double Word2VecVersionNumber;

//! Project version string for Word2Vec.
FOUNDATION_EXPORT const unsigned char Word2VecVersionString[];

// In this header, you should import all the public headers of your framework using statements like #import <Word2Vec/PublicHeader.h>


@interface Word2Vec : NSObject

@property (nonatomic, strong, nonnull) NSURL * trainFile;
@property (nonatomic, strong, nonnull) NSURL * outputFile;
@property (nonatomic, strong, nullable) NSURL * saveVocabFile;
@property (nonatomic, strong, nullable) NSURL * readVocabFile;
@property (nonatomic, strong, nullable) NSNumber * wordVectorSize;
@property (nonatomic, strong, nullable) NSNumber * debug;
@property (nonatomic, strong, nullable) NSNumber * saveToBinary;
@property (nonatomic, strong, nullable) NSNumber * continuousBagOfWords;
@property (nonatomic, strong, nullable) NSNumber * startingLearningRate;
@property (nonatomic, strong, nullable) NSNumber * windowLength;
@property (nonatomic, strong, nullable) NSNumber * wordsOccurrenceThreshold;
@property (nonatomic, strong, nullable) NSNumber * hierarchicalSoftmax;
@property (nonatomic, strong, nullable) NSNumber * negativeExamples;
@property (nonatomic, strong, nullable) NSNumber * threads;
@property (nonatomic, strong, nullable) NSNumber * trainingIterations;
@property (nonatomic, strong, nullable) NSNumber * minCount;
@property (nonatomic, strong, nullable) NSNumber * classesNumber;

-(instancetype _Nonnull ) initWithTrainFile:(NSURL*_Nonnull)trainFile andOutputFile:(NSURL*_Nonnull)outputFile;

-(BOOL) train;

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)closestToWord:(NSString * _Nonnull) word
                                                    numberOfClosest:(NSNumber * _Nullable) numberOfClosest;

- (NSDictionary <NSString *, NSNumber *>  * _Nullable)analogyToPhrase:(NSString * _Nonnull) phrase
                                                      numberOfClosest:(NSNumber * _Nullable) numberOfClosest;

@end
