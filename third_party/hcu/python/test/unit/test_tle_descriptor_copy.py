"""gfx938 TLE descriptor-copy regression, with independent raw-byte oracle."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@pytest.fixture(autouse=True)
def require_gfx938():
    if not torch.cuda.is_available():pytest.skip('HCU device required')
    t=triton.runtime.driver.active.get_current_target()
    if t.backend!='hip' or t.arch!='gfx938':pytest.skip('Validated gfx938 contract')

@triton.jit
def copy_transposed(x,o,H,W,S,BM:tl.constexpr,BN:tl.constexpr,DT:tl.constexpr,LAYOUT:tl.constexpr,FAIL:tl.constexpr=0):
    sm=tle.gpu.alloc((BM,BN),DT,layout=LAYOUT,scope=tle.gpu.smem,nv_mma_shared_layout=False)
    d=tl.make_tensor_descriptor(x,[H,W],[S,1],[BM,BN])
    rr=tl.broadcast_to(tl.arange(0,BM)[None,:],(BN,BM));cc=tl.broadcast_to(tl.arange(0,BN)[:,None],(BN,BM))
    p=tle.gpu.local_ptr(sm,(rr,cc))
    for it in range(3):
        if FAIL==1:tle.gpu.copy(d,sm,(BM,BN),offsets=(it*BM,0),mask=tl.full((BM,BN),True,tl.int1))
        elif FAIL==2:tle.gpu.copy(d,sm,(BM,BN),offsets=(it*BM,0),barrier=7)
        else:tle.gpu.copy(d,sm,(BM,BN),offsets=(it*BM,0))
        tl.store(o+it*BM*BN+cc*BM+rr,tl.load(p))

def shared8():
    return tle.gpu.swizzled_shared_layout(vectorSize=32,perPhase=2,maxPhase=2,order=[1,0],numCTAs=[1,1],numCTAsPerCGA=[1,1],numCTASplit=[1,1],numCTAOrder=[1,0])

@pytest.mark.parametrize('name,bm,bn,warps,stages,expect_mls',[
    ('f16',64,32,4,1,True),('i8',64,64,4,1,True),('fp8',64,64,4,1,True),
    ('f16',128,64,4,1,False),('i8',128,128,4,1,False),
    ('f16',64,32,8,1,False),('f16',64,32,4,2,False)])
def test_raw_bytes_and_fallback(name,bm,bn,warps,stages,expect_mls):
    td,dt={'f16':(torch.float16,tl.float16),'i8':(torch.int8,tl.int8),'fp8':(torch.float8_e4m3fn,tl.float8e4nv)}[name]
    h=3*bm+7;w=bn+9;s=bn+32;eb=torch.empty((),dtype=td).element_size()
    bits=((torch.arange((h*s+256)*eb,device='cuda',dtype=torch.int32)*37+61)%256).to(torch.uint8);before=bits.clone();x=bits.view(td)[128:]
    ob=torch.full(((3*bm*bn+256)*eb,),91,device='cuda',dtype=torch.uint8);o=ob.view(td)[128:-128]
    k=copy_transposed[(1,)](x,o,h,w,s,bm,bn,dt,None if name=='f16' else shared8(),num_warps=warps,num_stages=stages)
    torch.cuda.synchronize();host=before[128*eb:].cpu().reshape(-1,eb);rr=torch.arange(bm);cc=torch.arange(bn)
    ref=torch.stack([host[(it*bm+rr[None,:])*s+cc[:,None]] for it in range(3)])
    assert torch.equal(o.view(torch.uint8).cpu().reshape_as(ref),ref)
    assert bool((ob[:128*eb]==91).all()) and bool((ob[-128*eb:]==91).all())
    assert torch.equal(bits,before)
    assert ('matrix_load_' in k.asm['amdgcn'])==expect_mls

@pytest.mark.parametrize('fail,reason',[(1,'copy mask is only supported'),(2,'barrier')])
def test_unsupported_completion_contract(fail,reason):
    x=torch.zeros((256*128,),device='cuda',dtype=torch.float16);o=torch.empty((3*64*32,),device='cuda',dtype=torch.float16)
    with pytest.raises(Exception,match=reason):
        copy_transposed[(1,)](x,o,256,128,128,64,32,tl.float16,None,fail,num_warps=4,num_stages=1)
