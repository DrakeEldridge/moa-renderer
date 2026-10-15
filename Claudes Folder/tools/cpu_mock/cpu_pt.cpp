// CPU test harness: runs the real src/device/pathtracer.cu on the CPU through the OptiX mock
// in this folder (brute-force ray/triangle tests, no BVH). Slow, but lets path-tracer changes be
// checked visually without a GPU. Usage: cpu_pt <tall> <short> <launches> [background]
#include "device/pathtracer.cu"
#include "Scene.h"
#include "ImageIO.h"
#include <cstdio>
#include <vector>
#include <thread>
#include <cstring>
#include <string>
thread_local MockState g_mock;
static const moa::Scene* g_scene;
static bool hitTri(float3 o, float3 d, float3 a, float3 b, float3 c, float& t){
  using namespace moa; float3 e1=b-a,e2=c-a,p=cross(d,e2); float det=dot(e1,p); if(fabsf(det)<1e-12f) return false;
  float inv=1/det; float3 s=o-a; float u=dot(s,p)*inv; if(u<0||u>1) return false; float3 q=cross(s,e1); float v=dot(d,q)*inv; if(v<0||u+v>1) return false; t=dot(e2,q)*inv; return true; }
void mockTrace(OptixTraversableHandle, float3 o, float3 d, float tmin, float tmax, unsigned flags, unsigned, unsigned missIdx, unsigned* payload){
  float best=tmax; int bi=-1; const auto& V=g_scene->vertices;
  for(size_t i=0;i<g_scene->triangleCount();++i){ float t; if(hitTri(o,d,V[3*i],V[3*i+1],V[3*i+2],t) && t>tmin && t<best){best=t;bi=(int)i; if(flags&OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) break;} }
  MockState saved=g_mock; g_mock.payload=payload; g_mock.o=o; g_mock.d=d;
  if(bi<0){ if(missIdx==0) __miss__radiance(); else __miss__shadow(); }
  else if(!(flags&OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT)){ g_mock.prim=bi; g_mock.tmax=best; __closesthit__radiance(); }
  g_mock.payload=saved.payload; g_mock.o=saved.o; g_mock.d=saved.d; g_mock.prim=saved.prim; g_mock.tmax=saved.tmax; }
int main(int argc, char** argv){
  using namespace moa; CornellOptions opt; const float bg = argc>4 ? (float)atof(argv[4]) : 0.0f; if(argc>2){ parseMaterialType(argv[1],opt.tallBlock); parseMaterialType(argv[2],opt.shortBlock);} 
  Scene s=makeCornellBox(opt); g_scene=&s; const unsigned W=160,H=160, spl=8, launches=argc>3?atoi(argv[3]):8;
  std::vector<float4> beauty(W*H), albedo(W*H), normal(W*H);
  float aspect=1, th=tanf(0.5f*s.camera.verticalFovDegrees*kPi/180); float3 w=normalize(s.camera.lookAt-s.camera.eye), r=normalize(cross(w,s.camera.up)), up=normalize(cross(r,w));
  params.width=W; params.height=H; params.samplesPerLaunch=spl; params.maxDepth=8; params.maxSampleValue=4.0f; params.background=make_float3(bg,bg,bg); params.beauty=beauty.data(); params.albedo=albedo.data(); params.normal=normal.data();
  params.camera.eye=s.camera.eye; params.camera.w=w; params.camera.u=r*(th*aspect); params.camera.v=up*th; params.light=s.light;
  params.vertices=s.vertices.data(); params.triangleMaterial=s.triangleMaterial.data(); params.materials=s.materials.data();
  for(unsigned f=0; f<launches; ++f){ params.subframeIndex=f; std::vector<std::thread> th;
    for(unsigned t=0;t<16;t++) th.emplace_back([&,t]{ for(unsigned y=t;y<H;y+=16) for(unsigned x=0;x<W;x++){ g_mock.launchIdx=make_uint3(x,y,0); __raygen__render(); } });
    for(auto& x:th) x.join(); }
  double sum=0; for(auto& p:beauty) sum+=p.x+p.y+p.z; std::printf("mean radiance %.4f\n", sum/(3*W*H));
  std::string tag = argc>2? std::string(argv[1])+"_"+argv[2] : "default";
  writePng("cpu_"+tag+"_beauty.png",W,H,beauty,PngEncoding::Beauty); writePng("cpu_"+tag+"_normal.png",W,H,normal,PngEncoding::Normal); writePng("cpu_"+tag+"_albedo.png",W,H,albedo,PngEncoding::Albedo);
}
