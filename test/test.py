import torch
import my_oplib

A = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
B = torch.tensor([4, 5, 6], dtype=torch.float32).cuda()

C = torch.ops.my_oplib.vector_add(A, B)

print(C)

pic_in = torch.randint(0, 256, size=(600, 800, 3), dtype=torch.uint8)
pic_out_cpu = torch.ops.my_oplib.cvt_rgb_to_gray(pic_in)
print(pic_out_cpu)

pic_in = pic_in.cuda()
pic_out_cuda = torch.ops.my_oplib.cvt_rgb_to_gray(pic_in)
print(pic_out_cuda.cpu())