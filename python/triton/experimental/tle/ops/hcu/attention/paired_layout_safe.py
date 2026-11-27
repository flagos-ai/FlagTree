"""Checked GPU pack entry; initialize every padded key in each BN128 tile."""
import torch,triton
from .paired_layout import gpu_pack_v_paired

def pack_v_paired_gpu(inp,shape,out=None):
 if len(shape)!=4 or any(not isinstance(n,int) or n<1 for n in shape) or shape[3]!=128:raise ValueError('positive B,NK,H,D128 shape required')
 b,n,h,_=shape
 if inp.dtype not in (torch.uint8,torch.float8_e4m3fn) or not inp.is_cuda or not inp.is_contiguous() or inp.numel()!=b*n*h*128:raise ValueError('expected contiguous NHD E4M3FN or raw byte V on GPU')
 size=b*h*triton.cdiv(n,128)*16384
 if out is None:out=torch.empty(size,dtype=torch.uint8,device=inp.device)
 if out.dtype not in (torch.uint8,torch.float8_e4m3fn) or out.device!=inp.device or not out.is_contiguous() or out.numel()!=size:raise ValueError('wrong output dtype/device/size/contiguity')
 if max(inp.data_ptr(),out.data_ptr())<min(inp.data_ptr()+inp.numel(),out.data_ptr()+size):raise ValueError('pack output must not overlap input')
 # Launch both 64-row halves even when the second half is entirely padding.
 gpu_pack_v_paired[(triton.cdiv(n,128)*2,2,b*h)](inp.view(torch.uint8),out.view(torch.uint8),n,h,num_warps=4,num_stages=1)
 return out.view(torch.float8_e4m3fn)
