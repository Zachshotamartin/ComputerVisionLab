// Offline fixture evaluation of the native app's Vision settings; never opens a camera.
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>
#import "../native/macos/RectangleEdges.h"
int main(int argc, const char * argv[]) { @autoreleasepool {
 if(argc!=3)return 2;
 NSString *directory=[NSString stringWithUTF8String:argv[1]];
 NSDictionary *manifest=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"manifest.json"]] options:0 error:nil];
 NSMutableArray *records=[NSMutableArray array];
 for(NSDictionary *item in manifest[@"cases"]) {
  if(![item[@"kind"] isEqual:@"rectangle"])continue;
  NSURL *url=[NSURL fileURLWithPath:[directory stringByAppendingPathComponent:item[@"file"]]];
  VNDetectRectanglesRequest *request=[VNDetectRectanglesRequest new];request.maximumObservations=6;request.minimumConfidence=.65;request.minimumSize=.08;request.minimumAspectRatio=.2;
  NSError *error=nil;VNImageRequestHandler *handler=[[VNImageRequestHandler alloc] initWithURL:url options:@{}];[handler performRequests:@[request] error:&error];
  CGImageSourceRef source=CGImageSourceCreateWithURL((__bridge CFURLRef)url,NULL);CGImageRef image=CGImageSourceCreateImageAtIndex(source,0,NULL);
  size_t width=CGImageGetWidth(image),height=CGImageGetHeight(image),stride=width*4;uint8_t *pixels=calloc(height,stride);CGColorSpaceRef color=CGColorSpaceCreateDeviceRGB();
  CGContextRef context=CGBitmapContextCreate(pixels,width,height,8,stride,color,kCGImageAlphaPremultipliedFirst|kCGBitmapByteOrder32Little);CGContextDrawImage(context,CGRectMake(0,0,width,height),image);
  NSMutableArray *boxes=[NSMutableArray array];
  for(VNRectangleObservation *rect in request.results){if(!CVHasStraightEdges(rect,pixels,width,height,stride))continue;CGRect b=rect.boundingBox;[boxes addObject:@{ @"x":@(b.origin.x*320),@"y":@((1-b.origin.y-b.size.height)*240),@"width":@(b.size.width*320),@"height":@(b.size.height*240),@"score":@(rect.confidence)}];}
  CGContextRelease(context);CGColorSpaceRelease(color);CGImageRelease(image);CFRelease(source);free(pixels);
  [records addObject:@{@"file":item[@"file"],@"truth":item[@"truth"],@"actual":boxes,@"error":error?error.localizedDescription:@""}];
 }
 NSData *json=[NSJSONSerialization dataWithJSONObject:records options:NSJSONWritingPrettyPrinted error:nil];[json writeToFile:[NSString stringWithUTF8String:argv[2]] atomically:YES];
 }return 0;}
