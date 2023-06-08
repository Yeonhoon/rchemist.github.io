install.packages("torch")

library(torch)

t1 = torch::torch_tensor(1)
t1$dtype

t1$device

t2 = t1$to(dtype = torch_int())
t2$dtype

t3 = t1$view(c(1,1))
t3$shape

# torch_tensor array와 R의 array 생성방식 다름
torch_tensor(1:5, dtype=torch_int())

torch_tensor(matrix(1:9, nrow=3))

torch_tensor(array(1:24, dim = c(4,3,2)))

array(1:24, c(4,3,2))
torch_tensor(array(1:24, dim=c(2,4,3)))

torch_randn(3,3)
torch_rand(3,3)
torch_zeros(3,3)
torch_ones(3,3)
torch_eye(3,3)
torch_diag(c(1,3,5))
torch_tensor(JohnsonJohnson,)

class(JohnsonJohnson)
# factor: matrix 변환 시 character, numeric 변환 필요.
orange = Orange |> mutate(Tree= as.numeric(Tree)) |> as.matrix()
torch_tensor(as.matrix(orange), dtype = torch_int())

install.packages("modeldata")
library(modeldata)
?okc
modeldata::ames
data("stackoverflow")
stackoverflow
# character: as.factor -> as.numeric

as.factor(stackoverflow$Country) |> as.numeric() |> 
  torch_tensor()

torch_tensor(c(1,NA,3))


## Operation on tensors
t1 = torch_tensor(c(1,2))
t2 = torch_tensor(c(3,4))

torch_add(t1,t2) # no reference change
t1$add(t2) # no reference change
t1$add_(t2) # reference change
t1

# torch tensor: Not distinguish between row vectors and column vectors
t1 = torch_tensor(1:3)
t2 = torch_tensor(4:6)
t1$dot(t2)
t1$matmul(t2)
t1
t3 = torch_tensor(matrix(1:12, nrow=3, byrow=T))
t3
t1$matmul(t3)
torch_multiply(t1,t2)
t1$mul(t2)

## Summary operations
m = outer(1:3, 1:6)
sum(m)
apply(m,1,sum)
apply(m,2,sum)

t = torch_outer(torch_tensor(1:3), torch_tensor(1:6))
t$sum()
t$sum(dim=1) # column
t$sum(dim=2) # row

t = torch_randn(4,3,2)
t$mean()
t$mean(1)
t[1][1,]
t$mean(dim = 1) # 매트릭스 차원들의 값의 평균 계산
t$mean(dim = 2) # 매트릭스의 열 방향 계산
mean(c(-1.0524, 1.2195, 0.0897))

t$mean(dim=3) # 매트릭스의 행 방향 계산
mean(c(-0.3473, 0.2372, 0.1848))
t$mean(dim=c(1,2))


t$max(dim=1)


# Accessing parts of tensors

t = torch_tensor(matrix(1:9, ncol=3, byrow = T))
t[1,,drop=F]

t = torch_rand(3,3,3)
t[1:2, 2:3, c(1,3)]

t = torch_tensor(matrix(1:4, ncol=2, byrow=T))
t[-1,-1]

t = torch_tensor(matrix(1:20, nrow=2, byrow=T))
t[,1:8:2]
t[1,..,2:10]


## Reshape tensors

t = torch_zeros(24)
print(t, n=3)

t2 = t$view(c(2,12))
t2

# same address even though form changed
t$storage()$data_ptr()
t2$storage()$data_ptr()

## Broadcasting

t1 = torch_randn(3,5)
t1 *.5


t1 = torch_tensor(matrrix(1:15, ncol=5, byrow=T))
t2 = torch_tensor(matrix(1:5, ncol=5, byrow=T))
t3 = torch_tensor(1:5)
t1$shape
t2$shape
t1$add(t2)
t1$add(t3)



