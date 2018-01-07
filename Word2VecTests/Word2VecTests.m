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
    NSURL* binURL = [[NSBundle mainBundle] URLForResource:@"out.bin" withExtension:nil];
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

- (void)testExample {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
