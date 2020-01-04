"""
The main module.
"""
import torch

def run_tests():
    """
    Run tests.
    """
    s_test = (20, 100)
    test = torch.randn(*s_test)
    print(test.size())

if __name__ == '__main__':
    run_tests()