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


@end
