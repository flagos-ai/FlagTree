// Ordinary host decode+GPU greedy+CPU return. Persistent PID across three rounds.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <time.h>

static uint64_t now_ns() { timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec; }
static void check(bool b,const char *s){if(!b)throw std::runtime_error(s);}
static void bytes(const std::string&p,const void*v,size_t n){std::ofstream f(p,std::ios::binary);check(bool(f),"open raw");f.write((const char*)v,n);check(bool(f),"write raw");}
static void array(std::ostream&o,const std::vector<int>&v){o<<'[';for(size_t i=0;i<v.size();i++)o<<(i?",":"")<<v[i];o<<']';}
static void logs(enum ggml_log_level level,const char*t,void*){if(level!=GGML_LOG_LEVEL_DEBUG)std::cerr<<t;}

struct Session {
 llama_model*m=nullptr;llama_context*c=nullptr;llama_sampler*s=nullptr;llama_batch b{};
 int vocab=0,steps=0;std::vector<llama_token>prompt;std::vector<int>golden;
 Session(const char*checkpoint,const std::string&scenario,const char*goldfile){
  steps=scenario=="synthetic32"?32:64;
  auto mp=llama_model_default_params();mp.n_gpu_layers=-1;mp.use_extra_bufts=false;
  m=llama_model_load_from_file(checkpoint,mp);check(m,"model load");check(llama_model_n_layer(m)==64,"64 layers");
  auto*v=llama_model_get_vocab(m);vocab=llama_vocab_n_tokens(v);check(vocab==248320,"vocab");
  if(scenario=="chinese64"){
   std::string p="请用中文解释为什么量化模型的单 token decode 通常受显存带宽限制。";
   int n=-llama_tokenize(v,p.data(),p.size(),nullptr,0,true,true);prompt.resize(n);
   check(llama_tokenize(v,p.data(),p.size(),prompt.data(),n,true,true)==n&&n==17,"prompt protocol");
  }
  {std::ifstream f(goldfile);int t;while(f>>t)golden.push_back(t);check((int)golden.size()==steps,"fresh golden size");}
  auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=cp.n_ubatch=1;cp.n_threads=cp.n_threads_batch=8;
  cp.type_k=cp.type_v=GGML_TYPE_BF16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.no_perf=true;
  s=llama_sampler_chain_init(llama_sampler_chain_default_params());llama_sampler_chain_add(s,llama_sampler_init_greedy());
  llama_sampler_seq_config sc{0,s};cp.samplers=&sc;cp.n_samplers=1;c=llama_init_from_model(m,cp);check(c,"context");
  b=llama_batch_init(1,0,1);b.n_tokens=1;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
 }
 ~Session(){if(c){llama_synchronize(c);llama_batch_free(b);llama_free(c);}if(s)llama_sampler_free(s);if(m)llama_model_free(m);}
 void reset(){llama_memory_clear(llama_get_memory(c),true);llama_sampler_reset(s);llama_synchronize(c);check(llama_memory_seq_pos_max(llama_get_memory(c),0)==-1,"reset cursor");}
 void input(int token,int step){b.token[0]=token;b.pos[0]=step;}
 int decode(){check(llama_decode(c,b)==0,"decode");auto t=llama_get_sampled_token_ith(c,-1);check(t!=LLAMA_TOKEN_NULL,"GPU greedy missing");return t;}
 std::vector<uint8_t> state(const std::string&out){
  llama_synchronize(c);size_t n=llama_state_get_size(c);std::vector<uint8_t>guarded(n+256,0xA5);
  size_t got=llama_state_get_data(c,guarded.data()+128,n);check(got==n,"state serialization length");
  check(std::all_of(guarded.begin(),guarded.begin()+128,[](uint8_t x){return x==0xA5;})&&std::all_of(guarded.end()-128,guarded.end(),[](uint8_t x){return x==0xA5;}),"state snapshot caller guards");
  std::vector<uint8_t>v(guarded.begin()+128,guarded.end()-128);bytes(out,v.data(),v.size());return v;
 }
 std::vector<int> num(const std::string&out,const std::string&tag,const std::vector<int>*expected=nullptr,const std::vector<float>*expected_logits=nullptr,std::vector<float>*returned=nullptr){
  check(llama_set_sampler(c,0,nullptr),"disable sampler for full-logit diagnostics");reset();
  std::vector<int>tokens;std::vector<float>logits;logits.reserve(size_t(steps)*vocab);int token=1;
  for(int step=0;step<steps;step++){
   if(step<(int)prompt.size())token=prompt[step];input(token,step);check(llama_decode(c,b)==0,"NUM decode");llama_synchronize(c);
   const float*l=llama_get_logits_ith(c,-1);check(l,"NUM logits");
   for(int j=0;j<vocab;j++)check(std::isfinite(l[j]),"NUM nonfinite logits");
   token=std::max_element(l,l+vocab)-l;tokens.push_back(token);logits.insert(logits.end(),l,l+vocab);
   check(llama_memory_seq_pos_max(llama_get_memory(c),0)==step,"NUM cursor");
  }
  if(expected)check(tokens==*expected,"NUM own-token trajectory changed");
  if(expected_logits)check(logits.size()==expected_logits->size()&&std::memcmp(logits.data(),expected_logits->data(),logits.size()*sizeof(float))==0,"NUM full logits bitwise changed");
  bytes(out+"/"+tag+"-logits.f32",logits.data(),logits.size()*sizeof(float));state(out+"/"+tag+"-state.bin");
  if(returned)*returned=logits;
  // Backend chains may be initialized only once by the upstream API.
  // Diagnostics removed the old chain from the context; use a new identical
  // greedy chain when restoring sampling, outside every timed region.
  llama_sampler_free(s);s=llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(s,llama_sampler_init_greedy());
  check(llama_set_sampler(c,0,s),"restore ordinary GPU greedy");reset();return tokens;
 }
 void run(int round,const std::string&out){
  std::vector<float>base_logits;auto tokens=num(out,"pre",nullptr,nullptr,&base_logits);num(out,"pre-reset",&tokens,&base_logits);
  int first=std::max(0,int(prompt.size())-1);check(std::equal(tokens.begin()+first,tokens.begin()+first+10,golden.begin()+first),"NUM fresh golden10 failure");
  reset();std::vector<int>warm;int token=1;
  for(int step=0;step<steps;step++){if(step<(int)prompt.size())token=prompt[step];input(token,step);token=decode();warm.push_back(token);}
  check(warm==tokens,"GPU greedy versus NUM own trajectory");auto pre=state(out+"/warmup-endpoint-state.bin");
  reset();std::vector<uint64_t>latency;std::vector<int>actual;token=1;auto start=now_ns();
  for(int step=0;step<steps;step++){
   if(step<(int)prompt.size())token=prompt[step];input(token,step);
   auto before=now_ns();token=decode();auto after=now_ns();latency.push_back(after-before);actual.push_back(token);
  }
  auto end=now_ns();
  // Retain ordinary raw before post-timing checks. No full-logit fetch in timer.
  {std::ofstream f(out+"/raw.json");f<<"{\"round\":"<<round<<",\"pid\":"<<getpid()<<",\"start_ns\":"<<start<<",\"end_ns\":"<<end<<",\"steps\":"<<steps<<",\"latency_ns\":[";for(size_t i=0;i<latency.size();i++)f<<(i?",":"")<<latency[i];f<<"],\"tokens\":";array(f,actual);f<<",\"profiler_attached\":false,\"quantization_inside\":true}\n";}
  check(actual==tokens,"timed own trajectory mismatch");auto post=state(out+"/post-timing-endpoint-state.bin");check(pre==post,"post-timing state bitwise mismatch");
  num(out,"post",&tokens,&base_logits);
  {std::ofstream f(out+"/result.json");f<<"{\"status\":\"ROUND_PASS\",\"NUM_pre\":\"PASS\",\"NUM_post\":\"PASS\",\"round\":"<<round<<",\"pid\":"<<getpid()<<",\"finite\":true,\"bitwise\":true,\"guard_pass\":true,\"guard_scope\":\"caller-owned serialized-state redzones; upstream internal allocator guards not exposed\",\"prompt_token_count\":"<<prompt.size()<<",\"tokens\":";array(f,tokens);f<<"}\n";}
 }
};

int main(int argc,char**argv){
 if(argc!=4)return 2;
 try{
  llama_log_set(logs,nullptr);llama_backend_init();std::cout<<"{\"event\":\"READY\",\"pid\":"<<getpid()<<"}"<<std::endl;
  std::string line;
  while(std::getline(std::cin,line)){
   if(line=="EXIT")break;std::istringstream in(line);int round;std::string out;in>>round>>out;check(bool(in),"command");
   try{Session s(argv[1],argv[2],argv[3]);s.run(round,out);}
   catch(const std::exception&e){std::ofstream f(out+"/failure.txt");f<<e.what()<<'\n';throw;}
   std::cout<<"{\"event\":\"DONE\",\"round\":"<<round<<",\"pid\":"<<getpid()<<"}"<<std::endl;
  }
  llama_backend_free();return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;std::cout<<"{\"event\":\"FAIL\"}"<<std::endl;return 1;}
}

extern "C" int r031_llama_run(const char *checkpoint, const char *scenario,
                             const char *golden, int round, const char *out) {
 try {
  llama_log_set(logs, nullptr); llama_backend_init();
  { Session session(checkpoint, scenario, golden); session.run(round, out); }
  llama_backend_free(); return 0;
 } catch(const std::exception &e) {
  std::ofstream f(std::string(out)+"/failure.txt"); f << e.what() << '\n'; return 1;
 }
}
