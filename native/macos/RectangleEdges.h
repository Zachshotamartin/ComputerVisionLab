#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#include <math.h>

// Vision also proposes bounding rectangles around curved shapes. Require
// sustained contrast along each proposed straight side before drawing it.
static inline double CVLuminance(const uint8_t *pixels, size_t width, size_t height, size_t stride, double x, double y) {
    size_t ix=(size_t)fmax(0,fmin(width-1,round(x))), iy=(size_t)fmax(0,fmin(height-1,round(y)));
    const uint8_t *p=pixels+iy*stride+ix*4;
    return .0722*p[0]+.7152*p[1]+.2126*p[2]; // BGRA
}
static inline BOOL CVHasStraightEdges(VNRectangleObservation *rectangle, const uint8_t *pixels, size_t width, size_t height, size_t stride) {
    if (!pixels || !width || !height) return NO;
    CGPoint normalized[]={rectangle.topLeft,rectangle.topRight,rectangle.bottomRight,rectangle.bottomLeft}, corners[4];
    for(int i=0;i<4;i++)corners[i]=CGPointMake(normalized[i].x*width,(1-normalized[i].y)*height);
    for(int i=0;i<4;i++){
        CGPoint a=corners[i],b=corners[(i+1)%4];double dx=b.x-a.x,dy=b.y-a.y,length=hypot(dx,dy);
        if(length<16)return NO;
        double nx=-dy/length,ny=dx/length;int supported=0,count=40;
        for(int j=0;j<count;j++){
            double t=.08+.84*(j+.5)/count,x=a.x+t*dx,y=a.y+t*dy,best=0;
            for(int offset=-2;offset<=2;offset++){
                double px=x+offset*nx,py=y+offset*ny;
                double contrast=fabs(CVLuminance(pixels,width,height,stride,px+4*nx,py+4*ny)-CVLuminance(pixels,width,height,stride,px-4*nx,py-4*ny));
                best=fmax(best,contrast);
            }
            if(best>=8)supported++;
        }
        if(supported<count*.65)return NO;
    }
    return YES;
}
