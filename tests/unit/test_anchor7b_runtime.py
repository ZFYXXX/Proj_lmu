import copy
from contextlib import nullcontext
import json

import pytest
import torch

from mllm_comp.analysis.common.anchor7b_runtime import (
    AnchorAdapterSpec,create_anchor_adapter,global_batch_indices,save_adapter,load_adapter,identify_checkpoint,
)


def test_global_batch_has_no_duplicates_and_preserves_serial_order():
    visited=[]
    for step in range(320):
        parts=[global_batch_indices(step,world_size=2,accumulation=4,rank=r) for r in range(2)]
        assert sorted(parts[0]+parts[1])==list(range(step*8,(step+1)*8))
        visited.extend(parts[0]+parts[1])
    assert sorted(visited)==list(range(2560))
    with pytest.raises(ValueError):
        global_batch_indices(0,world_size=2,accumulation=4,rank=2)


def test_3584_zero_init_shape_and_parameter_count():
    module=create_anchor_adapter(AnchorAdapterSpec(3584,256))
    values=torch.randn(128,3584).bfloat16()
    result=module({"tokens":values})
    assert torch.equal(values,result["tokens"])
    assert sum(p.numel() for p in module.parameters())==1846016
    result["tokens"].float().square().mean().backward()
    assert module.adapter[-1].weight.grad.abs().sum()>0


def test_checkpoint_identity_and_next_optimizer_update(tmp_path):
    identity={"hidden_size":24,"identity_sha256":"toy-pinned-base"}
    torch.manual_seed(1)
    module=create_anchor_adapter(AnchorAdapterSpec(24,8))
    optimizer=torch.optim.AdamW(module.parameters(),lr=.0002)
    base={"tokens":torch.randn(128,24)}
    def step(m,o):
        m(base)["tokens"].square().mean().backward()
        o.step();o.zero_grad(set_to_none=True)
    step(module,optimizer)
    save_adapter(module,tmp_path,identity)
    restored=load_adapter(tmp_path,identity,"cpu")
    other=torch.optim.AdamW(restored.parameters(),lr=.0002)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(module,optimizer);step(restored,other)
    assert all(torch.equal(a,b) for a,b in zip(module.parameters(),restored.parameters()))
    with pytest.raises(ValueError,match="another base"):
        load_adapter(tmp_path,{"hidden_size":24,"identity_sha256":"wrong-base"},"cpu")


def test_reject_3b_checkpoint(tmp_path):
    (tmp_path/"config.json").write_text(json.dumps({"model_type":"qwen2_5_vl","hidden_size":2048,"num_hidden_layers":36}))
    with pytest.raises(ValueError,match="7B"):
        identify_checkpoint(tmp_path)


def _cpu_ddp_worker(rank, rendezvous, folder):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=f"file://{rendezvous}",rank=rank,world_size=2)
    torch.manual_seed(3)
    module=create_anchor_adapter(AnchorAdapterSpec(12,4))
    ddp=DDP(module)
    optimizer=torch.optim.SGD(module.parameters(),lr=.01)
    values=torch.randn(8,3,12,generator=torch.Generator().manual_seed(7))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for micro,index in enumerate(global_batch_indices(step,world_size=2,accumulation=2,rank=rank)):
            with (nullcontext() if micro==1 else ddp.no_sync()):
                (ddp({"tokens":values[index]})["tokens"].square().mean()/2).backward()
        optimizer.step()
    torch.save(module.state_dict(),f"{folder}/rank_{rank}.pt")
    dist.destroy_process_group()


def test_two_rank_accumulation_matches_serial_global_batch(tmp_path):
    torch.multiprocessing.spawn(_cpu_ddp_worker,args=(str(tmp_path/"gloo_init"),str(tmp_path)),nprocs=2,join=True)
    torch.manual_seed(3)
    module=create_anchor_adapter(AnchorAdapterSpec(12,4))
    optimizer=torch.optim.SGD(module.parameters(),lr=.01)
    values=torch.randn(8,3,12,generator=torch.Generator().manual_seed(7))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for index in range(step*4,(step+1)*4):
            (module({"tokens":values[index]})["tokens"].square().mean()/4).backward()
        optimizer.step()
    first=torch.load(tmp_path/"rank_0.pt",weights_only=True)
    second=torch.load(tmp_path/"rank_1.pt",weights_only=True)
    for name,value in module.state_dict().items():
        assert torch.equal(first[name],second[name])
        assert torch.allclose(first[name],value,atol=1e-7,rtol=1e-6)
