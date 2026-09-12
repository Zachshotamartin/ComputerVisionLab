#import "ViewController.h"
@import AVFoundation;
@import Vision;
@import QuartzCore;

@interface ViewController () <AVCaptureVideoDataOutputSampleBufferDelegate>
@property AVCaptureSession *session;
@property AVCaptureVideoPreviewLayer *preview;
@property CAShapeLayer *outlines;
@property NSView *cameraView;
@property NSTextField *status;
@property NSButton *toggle;
@property dispatch_queue_t captureQueue;
@property NSUInteger frameIndex;
@property (atomic) BOOL requestedRunning;
@end

@implementation ViewController
- (void)loadView {
    self.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 1000, 700)];
    self.view.wantsLayer = YES;
    self.view.layer.backgroundColor = [NSColor colorWithRed:.08 green:.14 blue:.12 alpha:1].CGColor;
    self.cameraView = [[NSView alloc] initWithFrame:NSMakeRect(20, 76, 960, 604)];
    self.cameraView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.cameraView.wantsLayer = YES;
    [self.view addSubview:self.cameraView];
    self.toggle = [NSButton buttonWithTitle:@"Start camera" target:self action:@selector(toggleCamera:)];
    self.toggle.frame = NSMakeRect(20, 22, 150, 32);
    [self.view addSubview:self.toggle];
    self.status = [NSTextField wrappingLabelWithString:@"Rectangle tracking · camera starts only when you choose Start camera."];
    self.status.frame = NSMakeRect(190, 18, 780, 40);
    self.status.autoresizingMask = NSViewWidthSizable;
    self.status.textColor = NSColor.secondaryLabelColor;
    [self.view addSubview:self.status];
    self.captureQueue = dispatch_queue_create("visionlab.camera", DISPATCH_QUEUE_SERIAL);
    self.outlines = [CAShapeLayer layer];
    self.outlines.strokeColor = [NSColor colorWithRed:.76 green:.9 blue:.63 alpha:1].CGColor;
    self.outlines.fillColor = NSColor.clearColor.CGColor;
    self.outlines.lineWidth = 3;
}

- (void)viewDidLayout {
    [super viewDidLayout];
    self.preview.frame = self.cameraView.bounds;
    self.outlines.frame = self.cameraView.bounds;
}

- (void)toggleCamera:(id)sender {
    if (self.requestedRunning) { [self stopCamera]; return; }
    self.requestedRunning = YES;
    self.toggle.title = @"Cancel";
    self.status.stringValue = @"Waiting for camera access…";
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (!self.requestedRunning) return;
            if (!granted) {
                [self stopCamera];
                self.status.stringValue = @"Camera access is disabled. Enable it in System Settings → Privacy & Security → Camera.";
                return;
            }
            [self startCamera];
        });
    }];
}

- (void)startCamera {
    if (!self.session) {
        AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        NSError *error = nil;
        AVCaptureDeviceInput *input = device ? [AVCaptureDeviceInput deviceInputWithDevice:device error:&error] : nil;
        if (!input) {
            [self stopCamera];
            self.status.stringValue = error.localizedDescription ?: @"No camera is available.";
            return;
        }
        AVCaptureSession *session = [AVCaptureSession new];
        session.sessionPreset = AVCaptureSessionPreset640x480;
        AVCaptureVideoDataOutput *output = [AVCaptureVideoDataOutput new];
        output.alwaysDiscardsLateVideoFrames = YES;
        output.videoSettings = @{(NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)};
        [output setSampleBufferDelegate:self queue:self.captureQueue];
        if (![session canAddInput:input] || ![session canAddOutput:output]) {
            [self stopCamera];
            self.status.stringValue = @"The camera could not be configured.";
            return;
        }
        [session addInput:input]; [session addOutput:output];
        self.session = session;
        self.preview = [AVCaptureVideoPreviewLayer layerWithSession:session];
        self.preview.videoGravity = AVLayerVideoGravityResizeAspect;
        self.preview.frame = self.cameraView.bounds;
        [self.cameraView.layer addSublayer:self.preview];
        [self.cameraView.layer addSublayer:self.outlines];
    }
    self.toggle.title = @"Stop camera";
    self.status.stringValue = @"Find a book, card, or rectangular surface. Frames are processed on this Mac.";
    dispatch_async(self.captureQueue, ^{ if (self.requestedRunning) [self.session startRunning]; });
}

- (void)stopCamera {
    self.requestedRunning = NO;
    self.toggle.title = @"Start camera";
    self.status.stringValue = @"Camera stopped.";
    self.outlines.path = nil;
    AVCaptureSession *session = self.session;
    dispatch_async(self.captureQueue, ^{ [session stopRunning]; });
}

- (void)viewDidDisappear { [super viewDidDisappear]; [self stopCamera]; }

- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)buffer fromConnection:(AVCaptureConnection *)connection {
    if (!self.requestedRunning || (++self.frameIndex % 4)) return;
    VNDetectRectanglesRequest *request = [VNDetectRectanglesRequest new];
    request.maximumObservations = 6;
    request.minimumConfidence = .65;
    request.minimumSize = .08;
    request.minimumAspectRatio = .2;
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCMSampleBuffer:buffer orientation:kCGImagePropertyOrientationUp options:@{}];
    NSError *error = nil;
    if (![handler performRequests:@[request] error:&error]) {
        dispatch_async(dispatch_get_main_queue(), ^{ if (self.requestedRunning) self.status.stringValue = error.localizedDescription; });
        return;
    }
    NSArray<VNRectangleObservation *> *rectangles = request.results;
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!self.requestedRunning) return;
        CGMutablePathRef path = CGPathCreateMutable();
        for (VNRectangleObservation *rectangle in rectangles) {
            CGPoint corners[] = {rectangle.topLeft, rectangle.topRight, rectangle.bottomRight, rectangle.bottomLeft};
            for (int index = 0; index < 4; index++) {
                CGPoint point = [self.preview pointForCaptureDevicePointOfInterest:CGPointMake(corners[index].x, 1 - corners[index].y)];
                if (index == 0) CGPathMoveToPoint(path, NULL, point.x, point.y);
                else CGPathAddLineToPoint(path, NULL, point.x, point.y);
            }
            CGPathCloseSubpath(path);
        }
        self.outlines.path = path;
        CGPathRelease(path);
        self.status.stringValue = [NSString stringWithFormat:@"%lu rectangle%@ · 2D outline tracking · frames remain on this Mac", (unsigned long)rectangles.count, rectangles.count == 1 ? @"" : @"s"];
    });
}
@end

@interface CVAppDelegate : NSObject <NSApplicationDelegate>
@property NSWindow *window;
@end
@implementation CVAppDelegate
- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    self.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 1000, 700) styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable backing:NSBackingStoreBuffered defer:NO];
    self.window.title = @"Vision Lab — Rectangle Tracker";
    self.window.minSize = NSMakeSize(640, 480);
    self.window.contentViewController = [ViewController new];
    [self.window center]; [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender { return YES; }
@end

int CVApplicationMain(int argc, const char **argv) {
    @autoreleasepool {
        NSApplication *application = [NSApplication sharedApplication];
        [application setActivationPolicy:NSApplicationActivationPolicyRegular];
        CVAppDelegate *delegate = [CVAppDelegate new];
        application.delegate = delegate;
        NSMenu *menu = [NSMenu new];
        NSMenuItem *item = [NSMenuItem new];
        NSMenu *applicationMenu = [NSMenu new];
        [applicationMenu addItemWithTitle:@"Quit Vision Lab" action:@selector(terminate:) keyEquivalent:@"q"];
        item.submenu = applicationMenu; [menu addItem:item]; application.mainMenu = menu;
        [application run];
    }
    return 0;
}
