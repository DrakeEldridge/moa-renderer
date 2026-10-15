// CPU test harness: runs the real src/device/pathtracer.cu on the CPU through the OptiX mock
// in this folder (brute-force ray/triangle tests, no BVH). Slow, but lets path-tracer changes be
// checked visually without a GPU. Usage: cpu_pt <tall> <short> <launches> [background]
#include "device/pathtracer.cu"
#include "Scene.h"
#include "ImageIO.h"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <thread>
#include <cstring>
#include <string>
thread_local MockState g_mock;
std::vector<MockTexture> g_mockTextures;
static const moa::Scene* g_scene;
static bool hitTri(float3 o, float3 d, float3 a, float3 b, float3 c, float& t, float2& bc){
  using namespace moa; float3 e1=b-a,e2=c-a,p=cross(d,e2); float det=dot(e1,p); if(fabsf(det)<1e-12f) return false;
  float inv=1/det; float3 s=o-a; float u=dot(s,p)*inv; if(u<0||u>1) return false; float3 q=cross(s,e1); float v=dot(d,q)*inv; if(v<0||u+v>1) return false; t=dot(e2,q)*inv; bc=make_float2(u,v); return true; }
// Small median-split BVH so the CPU test can handle real models (tens of thousands of triangles).
struct BvhNode { float3 lo, hi; int left, right, first, count; };
static std::vector<BvhNode> g_nodes; static std::vector<int> g_triIdx;
static float3 triCentroid(int t){ const auto& V=g_scene->positions; const uint3 i=g_scene->indices[t]; return (V[i.x]+V[i.y]+V[i.z])*(1.0f/3.0f); }
static int buildBvh(int first, int count){
  using namespace moa; BvhNode n; n.lo=make_float3(1e30f,1e30f,1e30f); n.hi=make_float3(-1e30f,-1e30f,-1e30f);
  const auto& V=g_scene->positions;
  for(int k=first;k<first+count;k++){ const uint3 i=g_scene->indices[g_triIdx[k]]; for(uint32_t vi: {i.x,i.y,i.z}){ const float3 p=V[vi];
    n.lo=make_float3(fminf(n.lo.x,p.x),fminf(n.lo.y,p.y),fminf(n.lo.z,p.z)); n.hi=make_float3(fmaxf(n.hi.x,p.x),fmaxf(n.hi.y,p.y),fmaxf(n.hi.z,p.z)); } }
  n.left=n.right=-1; n.first=first; n.count=count; const int id=(int)g_nodes.size(); g_nodes.push_back(n);
  if(count<=4) return id;
  float3 e=n.hi-n.lo; int axis = e.x>e.y&&e.x>e.z?0:(e.y>e.z?1:2);
  auto key=[&](int t){ float3 c=triCentroid(t); return axis==0?c.x:(axis==1?c.y:c.z); };
  int mid=first+count/2; std::nth_element(g_triIdx.begin()+first, g_triIdx.begin()+mid, g_triIdx.begin()+first+count, [&](int a,int b){return key(a)<key(b);});
  const int l=buildBvh(first, mid-first); const int r=buildBvh(mid, first+count-mid);
  g_nodes[id].left=l; g_nodes[id].right=r; g_nodes[id].count=0; return id; }
static bool hitBox(const BvhNode& n, float3 o, float3 inv, float tmax){
  float t0=0, t1=tmax; const float lo[3]={n.lo.x,n.lo.y,n.lo.z}, hi[3]={n.hi.x,n.hi.y,n.hi.z}, oo[3]={o.x,o.y,o.z}, ii[3]={inv.x,inv.y,inv.z};
  for(int a=0;a<3;a++){ float ta=(lo[a]-oo[a])*ii[a], tb=(hi[a]-oo[a])*ii[a]; if(ta>tb) std::swap(ta,tb); t0=fmaxf(t0,ta); t1=fminf(t1,tb); if(t0>t1) return false; } return true; }
void mockTrace(OptixTraversableHandle, float3 o, float3 d, float tmin, float tmax, unsigned flags, unsigned, unsigned missIdx, unsigned* payload){
  float best=tmax; int bi=-1; float2 bb=make_float2(0,0); const auto& V=g_scene->positions; const auto& I=g_scene->indices;
  const float3 inv=make_float3(1.0f/d.x,1.0f/d.y,1.0f/d.z); int stack[64]; int sp=0; stack[sp++]=0; bool done=false;
  while(sp && !done){ const BvhNode& n=g_nodes[stack[--sp]]; if(!hitBox(n,o,inv,best)) continue;
    if(n.count){ for(int k=n.first;k<n.first+n.count;k++){ const int i=g_triIdx[k]; float t; float2 bc;
        if(hitTri(o,d,V[I[i].x],V[I[i].y],V[I[i].z],t,bc) && t>tmin && t<best){best=t;bi=i;bb=bc; if(flags&OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT){done=true;break;}} } }
    else { stack[sp++]=n.left; stack[sp++]=n.right; } }
  MockState saved=g_mock; g_mock.payload=payload; g_mock.o=o; g_mock.d=d;
  if(bi<0){ if(missIdx==0) __miss__radiance(); else __miss__shadow(); }
  else if(!(flags&OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT)){ g_mock.prim=bi; g_mock.tmax=best; g_mock.bary=bb; __closesthit__radiance(); }
  g_mock.payload=saved.payload; g_mock.o=saved.o; g_mock.d=saved.d; g_mock.prim=saved.prim; g_mock.tmax=saved.tmax; g_mock.bary=saved.bary; }
int main(int argc, char** argv){
  using namespace moa; CornellOptions opt; const float bg = argc>4 ? (float)atof(argv[4]) : 0.0f; if(argc>2){ parseMaterialType(argv[1],opt.tallBlock); parseMaterialType(argv[2],opt.shortBlock);} 
  Mesh model; if(argc>5){ std::string err; if(!loadMesh(argv[5],model,err)){std::printf("%s\n",err.c_str()); return 1;} opt.model=&model; opt.modelMaterial=opt.tallBlock; { std::string log; opt.modelTextures=loadModelTextures(model,log); std::printf("%s", log.c_str()); } if(argc>6) opt.modelZUp=true; std::printf("model tris %zu\n", model.triangleCount()); }
  Scene s=makeCornellBox(opt); g_scene=&s;
  g_triIdx.resize(s.triangleCount()); for(size_t i=0;i<g_triIdx.size();++i) g_triIdx[i]=(int)i; buildBvh(0,(int)g_triIdx.size()); const unsigned W=160,H=160, spl=8, launches=argc>3?atoi(argv[3]):8;
  std::vector<float4> beauty(W*H), albedo(W*H), normal(W*H);
  float aspect=1, th=tanf(0.5f*s.camera.verticalFovDegrees*kPi/180); float3 w=normalize(s.camera.lookAt-s.camera.eye), r=normalize(cross(w,s.camera.up)), up=normalize(cross(r,w));
  params.width=W; params.height=H; params.samplesPerLaunch=spl; params.maxDepth=8; params.maxSampleValue=4.0f; params.background=make_float3(bg,bg,bg); params.beauty=beauty.data(); params.albedo=albedo.data(); params.normal=normal.data();
  params.camera.eye=s.camera.eye; params.camera.w=w; params.camera.u=r*(th*aspect); params.camera.v=up*th; params.light=s.light;
  params.positions=s.positions.data(); params.normals=s.normals.data(); params.indices=s.indices.data();
  params.texcoords=s.texcoords.data(); params.tangents=s.tangents.data();
  static std::vector<cudaTextureObject_t> texIds; for(size_t i=0;i<s.textures.size();++i){ const Image& im=*s.textures[i].image; g_mockTextures.push_back({im.width,im.height,im.rgba.data(),s.textures[i].srgb}); texIds.push_back(i); }
  if(texIds.empty()) texIds.push_back(0); params.textures=texIds.data(); params.triangleMaterial=s.triangleMaterial.data(); params.materials=s.materials.data();
  for(unsigned f=0; f<launches; ++f){ params.subframeIndex=f; std::vector<std::thread> th;
    for(unsigned t=0;t<16;t++) th.emplace_back([&,t]{ for(unsigned y=t;y<H;y+=16) for(unsigned x=0;x<W;x++){ g_mock.launchIdx=make_uint3(x,y,0); __raygen__render(); } });
    for(auto& x:th) x.join(); }
  double sum=0; for(auto& p:beauty) sum+=p.x+p.y+p.z; std::printf("mean radiance %.4f\n", sum/(3*W*H));
  std::string tag = argc>2? std::string(argv[1])+"_"+argv[2] : "default";
  writePng("cpu_"+tag+"_beauty.png",W,H,beauty,PngEncoding::Beauty); writePng("cpu_"+tag+"_normal.png",W,H,normal,PngEncoding::Normal); writePng("cpu_"+tag+"_albedo.png",W,H,albedo,PngEncoding::Albedo);
}
