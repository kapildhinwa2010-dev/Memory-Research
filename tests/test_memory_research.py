import pytest
import torch
from memory_research import MemoryResearch, create_memory_research

class TestMemoryResearch:
    def test_init(self):
        m = MemoryResearch(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = MemoryResearch(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_memory_research()
        assert isinstance(m, MemoryResearch)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
