from pathlib import Path
import json,subprocess,hashlib
A=Path(__file__).resolve().parent;W=A.parents[3]
R3=W/'.local/milestones/m000/revisions/r003/profiles/model/r003-fullmodel-ab'
src=W/'.local/agents/graph_profiler/014-r003_fullmodel_ab/llama_worker.cpp'
build=R3/'builds/llama-20261117T151150224404Z'
source=src.read_text()
# Session/run are copied byte-for-byte; only a host C ABI is appended.
bridge=source+'''
extern "C" int r031_llama_run(const char *checkpoint, const char *scenario,
                             const char *golden, int round, const char *out) {
 try {
  llama_log_set(logs, nullptr); llama_backend_init();
  { Session session(checkpoint, scenario, golden); session.run(round, out); }
  llama_backend_free(); return 0;
 } catch(const std::exception &e) {
  std::ofstream f(std::string(out)+"/failure.txt"); f << e.what() << '\\n'; return 1;
 }
}
'''
(A/'llama_bridge.cpp').write_text(bridge)
argv=['g++','-O3','-std=c++17','-shared','-fPIC','-I'+str(build/'source/include'),'-I'+str(build/'source/ggml/include'),str(A/'llama_bridge.cpp'),'-L'+str(build/'build/bin'),'-Wl,-rpath,'+str(build/'build/bin'),str(build/'build/bin/libllama.so.0.9.0'),str(build/'build/bin/libggml.so.0.29.0'),'-o',str(A/'llama_bridge.so')]
root=Path('/home/sunnycase/.cache/flagtree-cpp-qwen38/reference-src/llama.cpp-5ce06fc7')
argv[5]='-I'+str(root/'include');argv[6]='-I'+str(root/'ggml/include')
p=subprocess.run(argv,capture_output=True,text=True)
(A/'llama-bridge-build.json').write_text(json.dumps(dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),session_run_unchanged=True),indent=2)+'\n')
print(p.returncode,p.stderr,flush=True)
assert p.returncode==0
