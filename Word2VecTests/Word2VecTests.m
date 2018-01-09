//
//  Word2VecTests.m
//  Word2VecTests
//
//  Created by Adam Wulf on 1/6/18.
//

#import <XCTest/XCTest.h>
#import <Word2Vec/Word2Vec.h>

@interface Word2VecTests : XCTestCase

@end

@implementation Word2VecTests{
    Word2Vec* model;
}

- (void)setUp {
    [super setUp];
    model = [[Word2Vec alloc] init];
    NSURL* binURL = [[NSBundle bundleForClass:[self class]] URLForResource:@"out.bin" withExtension:nil];
    [model setOutputFile:binURL];
}

- (void)tearDown {
    NSArray<NSString *> * paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, true);
    NSString* documentsDirectory = paths[0];
    NSURL *url = [[NSURL fileURLWithPath:documentsDirectory] URLByAppendingPathComponent:@"out.bin"];
    BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:[url path]];
    if(exists){
        @try{
            [[NSFileManager defaultManager] removeItemAtPath:[url path] error:nil];
        }@catch(id e){
            // noop
        }
    }
    [super tearDown];
}

- (void) testTrain{
    NSArray<NSString *> * paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, true);
    NSString* documentsDirectory = paths[0];
    NSURL *gooURL = [[NSBundle bundleForClass:[self class]] URLForResource:@"pg2701" withExtension:@"txt"];
    XCTAssertNotNil(gooURL);
    
    NSURL *url = [[NSURL fileURLWithPath:documentsDirectory] URLByAppendingPathComponent:@"out.bin"];
    BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:[url path]];
    XCTAssertFalse(exists);
    
    [model setTrainFile:gooURL];
    [model setOutputFile:url];
    [model train];

    exists = [[NSFileManager defaultManager] fileExistsAtPath:[url path]];
    XCTAssertTrue(exists);
}

-(void) testPerformane {
    [self measureBlock:^{
        NSString *init_word = @"bird";
        NSArray<NSString*>* acc = @[init_word];
        NSMutableDictionary<NSString*, NSNumber*>*result = [[model closestToWord:init_word numberOfClosest:@10] mutableCopy];
        
        __block NSNumber* max;
        __block NSString* closest;
        [result enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull key, NSNumber * _Nonnull obj, BOOL * _Nonnull stop) {
            if([obj doubleValue] > [max doubleValue]){
                max = obj;
                closest = key;
            }
        }];
        
        acc  = [acc arrayByAddingObject:closest];

        for (int i=0; i<100; i++) {
            result = [[model closestToWord:closest numberOfClosest:@10] mutableCopy];
            for (int j=0; j<[result count]; j++) {
                max = nil;
                [result enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull key, NSNumber * _Nonnull obj, BOOL * _Nonnull stop) {
                    if([obj doubleValue] > [max doubleValue]){
                        max = obj;
                        closest = key;
                    }
                }];
                
                NSString* new_association = closest;
                
                if ([acc containsObject:new_association]) {
                    [result removeObjectForKey:new_association];
                } else {
                    acc = [acc arrayByAddingObject:new_association];
                    break;
                }
            }
        }
    }];
}

-(void) testDistance {
    NSDictionary* result = [model closestToWord:@"cat" numberOfClosest:@1];
    XCTAssertEqualObjects([[result allKeys] firstObject], @"dog");
}

-(void) testAnalog {
    NSDictionary* result = [model analogyToPhrase:@"man woman king" numberOfClosest:@1];
    XCTAssertEqualObjects([[result allKeys] firstObject], @"queen");
    
    result = [model analogyToPhrase:@"pet toy" numberOfClosest:@1];
    XCTAssertEqualObjects([[result allKeys] firstObject], @"eat");
}

-(void) testSimilarity {
    NSDictionary* result = [model wordSimilarity:@"green red purple turtle"];
    NSString* leastSimilar = nil;
    CGFloat minSimilar = 1;
    
    for (NSString* key in [result allKeys]) {
        if([result[key] floatValue] < minSimilar){
            leastSimilar = key;
            minSimilar = [result[key] floatValue];
        }
    }
    
    XCTAssertEqualObjects(leastSimilar, @"turtle");
}

-(void) testSimilarity2 {
    NSDictionary* result = [model wordSimilarity:@"king castle sword car"];
    NSString* leastSimilar = nil;
    CGFloat minSimilar = 1;
    
    for (NSString* key in [result allKeys]) {
        if([result[key] floatValue] < minSimilar){
            leastSimilar = key;
            minSimilar = [result[key] floatValue];
        }
    }
    
    XCTAssertEqualObjects(leastSimilar, @"car");
}

@end
