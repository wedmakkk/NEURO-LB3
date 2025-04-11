import torch
import random
tenz=torch.zeros(1,4,dtype=torch.int)
print(tenz)
tenz = torch.randint(1,10,(1,4))
print(tenz)
 
tenz = tenz.to(dtype=torch.float32)
tenz.requires_grad=True 
print(tenz)

stepen=tenz**2 # 7 вариант (нечетный) 
print(stepen)
x=random.randint(1,10)
print("рандомное число =",x)
proizv=stepen*x
print(proizv)
exponentaa=torch.exp(proizv)
print(exponentaa)
out=exponentaa.mean()
out.backward()  # Нахожу градиент
print(tenz.grad)