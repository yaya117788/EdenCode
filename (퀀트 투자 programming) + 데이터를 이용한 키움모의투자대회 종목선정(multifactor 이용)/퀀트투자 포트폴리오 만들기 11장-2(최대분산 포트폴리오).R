library(quantmod)
library(PerformanceAnalytics)
library(magrittr)
library(tidyr)
library(dplyr)
library(corrplot)
library(nloptr)
library(quadprog)
library(RiskPortfolios)
library(ggplot2)
''' 포트폴리오 변동성은 sum(ij [1-> n])(w(i)*w(j)*stdev(ij))로 표현할 수 있지만 다음과 같이 변화가 가능하다.
sum(i [1 ->n])(w(i)^2*stddev(i)^2) + sum(i [1 -> n], [i!=j -> n])(w(i)*w(j)*cor(ij)*stddev(i)*stddev(j)) 로 변경될 수 있다.
뒷부분에는 자산간 상관관계가 포함되어있다. 상관관계가 낮아질수록 포트폴리오의 변동성 또한 점차 낮아지는데 이를 '분산효과'라고 한다.
이런 분산효과의 정도를 측정하는 지표가 분산비율(DR:Diversification Ratio)입니다. 분산비율의 분자는 개별 변동성의 가중합이며, 분모는 포트폴리오의 변동성이다.
대부분 자산간 상관관계는 1보다 낮으며, 이로인해 포트폴리오 분산은 단순 가중합보다 작아져 분모가 작아지고 1보다 커지게 된다.
최대분산효과 포트폴리오는 분산효과가 최대로 되는, 즉, 분산비율이 최대가 되는 potforlio를구성하는 방법입니다.
'''

''' 목적함수는 분산비율을 최대화 하는데에 있는 반면 대부분의 최적화 프로그래밍은 목적함수를 
최소화하는 형태로 이루어집니다. 따라서 목적함수인 maxDR을 최소화하는 형태로 바꿀 필요가 있는데 세가지 방법이 있다.
1. Choueifaty Synthetic Asset Back-Transformation을 이용하는 방법
- 이방법은 먼저 목적함수를 (minW`CW의 합[여기서 C는 분산-공분산행렬이 아닌 상관과계행렬이용])으로 갖고 제약조건(총합1,비중최소 0이상)을 만족하는 자살별 비중을 구한후 구해진 비중을 각각의 표준편차로 나누어 주며 비중의 합이 1이 되도록 표준화해준다.
2. Duality를 이용하는 방법
- 목적함수는 최소분산 포트폴리오와 동일하며, 제약조건만 개별자산의 투자비중 최소 0이상과 개별 표준편차의 가중합이 1인조건으로 바꾼다. 이후 비중의 합이 1이되도록 표준화 시켜준다. EX > SUM(W(i)*stddev(i)) = 1
3. MIN (-)DR의 방법
- 표준화가 불필요하고 간단하게 목적함수의 형태를 바꾸면 된다.
'''

# solve.Qp() 함수를 이용한 최적화
''' 먼저 solve.Qp() 함수를 이용해 Duality 방법을 사용해본다. Duality 목적함수는 최소분산포트폴리오와 동일하며,
제약조건은 비중합이 1이고 각 자산비중이 0보다 크다는 것을 쓴다. 그래서 Amat, bvec 부분을 입력할때 이부분을 고려해야한다.
'''
Dmat = covmat
dvec = rep(0, 10)
Amat = t(rbind(sqrt(diag(covmat)), diag(10)))
bvec = c(1, rep(0, 10))
meq = 1
diag(covmat)
''' Amat과 bvec부분이 최소분산과 다른데 이는 첫째로
1행의 sqrt(diag(covmet))은 각 비중과 행렬곱이 되며 각 자산비중합 = 1이라는 것과 같으며 이는 등위제약조건을 의미한다. 
2행부터 마지막행까지는 모두 각 비중이 0보다 큰 조건을 의미한다.
즉, 행렬의 맨 왼쪽에 해당하는 Amat은 각 자산의 표준편차로 이루어진 벡터행렬과 1로이루어진 대각행렬로 구성되어있다.
먼저 diag(covmat)을 통해 분산-공분산행렬에서 대각부분(분산부분)만 뽑아와서 sqrt로 표준편차로 만든다. (개별 자산의 분산인  
σ(i,i)는σ(i)σ(i)ρ(1,1)형태로 쓸 수 있으며,ρ(1,1)=1을 적용하면 σ(i)^2와 같습니다 = 분산)
이후 diag(10)을 통해 만든 대각행렬과 행으로 묶어준 후 전치행렬을 입력합니다.
bvec은 행렬의 맨오른쪽과 같이 등위제약조건에 해당하는 1과 부등위 제약조건에 해당하는 0들로 구성되어있습니다. 차후에 표준화 과정을 거쳐야 하므로
Duality 방법에서는 개별자산의 투자비중이 1보다 작은 조건을 입력하지 않아도 된다.'''

result = solve.QP(Dmat, dvec, Amat, bvec, meq)

w = result$solution %>%
  round(., 4) %>%
  setNames(colnames(rets))

print(w)
''' 최적화를 수행한후 결과값을 보면 비중의 합이 1을 초과하게 된다. 정규화 과정을 통해 비중의 합이 1이되도록 
표준화를 해준다.'''
w = (w / sum(w)) %>%
  round(., 4)

print(w)

data.frame(w) %>%
  ggplot(aes(x = factor(rownames(.), levels = rownames(.)),
             y = w)) +
  geom_col() +
  geom_col() +
  xlab(NULL) + ylab(NULL)
## 그래프를 통해 비중을 확인해본다.



# optimalPortfolio() 함수를 이용한 최적화
'''이 함수를 통해선 최소분산과 같이 매우 간단하게 포트폴리오 구현가능하다.'''
w = optimalPortfolio(covmat,
                     control = list(type = 'maxdiv',
                                    constraint = 'lo')) %>%
  round(., 4)
'''type에 maxdiv(MAXIMUM diversification을 의미)를 입력해주며 제약조건에는 
투자비중이 0보다 큰 lo조건을 입력합니다. 해당함수의 코드를 확인해보면 최대분산포트폴리오 계산시
Min - DR 방법을 사용합니다.'''
print(w)



# 최소 최대비중 제약조건 추가
''' 최대분산효과 포트폴리오 역시 구석해 문제가 발생하며 모든 자산에 골고루 투자하기 위해
개별 투자 비중을 최소 5%, 최대 20%로 하는 제약조건을 추가하겠습니다.'''

''' Duality 방법에서는 목적함수인 min 1/2*w`*stddev*w와 제약조건에 맞게 해를 구한후 표준화를 시켰습니다. 따라서
비중의 최대 최소 제약조건은 단순히 lb <= w <= ub가 아닌 표준화 과정까지 고려해 적용해야합니다. 
ex> w(i)/ sum(w(i)) >= lb로 놓고 푼다. >  (-lb * e^T +I)w >= 0 으로 나온다. 
lb * e^T의 경우 계산하면 -lb로 이루어진 nxn 행렬이고 I는 항등행렬이니 대각부분만 1이 더해진다.'''

Dmat = covmat
dvec = rep(0, 10)
Alb = -rep(0.05, 10) %*% matrix(1, 1, 10) + diag(10)
Aub = rep(0.20, 10) %*% matrix(1, 1, 10) - diag(10)
''' Alb의 -rep(0.05,10)은 -lb부분, matrix(1,1,10)은 e^T부분, diag(10)부분은 I부분을 의미한다. 이는 최소비중 제약조건의 좌변과 동일하다.'''

Amat = t(rbind(sqrt(diag(covmat)), Alb, Aub))
bvec = c(1, rep(0, 10), rep(0, 10))
meq = 1

result = solve.QP(Dmat, dvec, Amat, bvec, meq)

w = result$solution 
w = (w / sum(w)) %>%
  round(., 4) %>%
  setNames(colnames(rets))

print(w)

data.frame(w) %>%
  ggplot(aes(x = factor(rownames(.), levels = rownames(.)),
             y = w)) +
  geom_col() +
  geom_hline(aes(yintercept = 0.05), color = 'red') +
  geom_hline(aes(yintercept = 0.20), color = 'red') +
  xlab(NULL) + ylab(NULL)


# 각각의 자산별로 다른 최대 최소 비중 제약조건을 추가시켜본다. (응용)
Dmat = covmat
dvec = rep(0, 10)
Alb = -c(0.10, 0.10, 0.05, 0.05, 0.10,
         0.10, 0.05, 0.05, 0.03, 0.03) %*%
  matrix(1, 1, 10) + diag(10)
Aub = c(0.25, 0.25, 0.20, 0.20, 0.20,
        0.20, 0.10, 0.10, 0.08, 0.08) %*%
  matrix(1, 1, 10) - diag(10)

Amat = t(rbind(sqrt(diag(covmat)), Alb, Aub))
bvec = c(1, rep(0, 10), rep(0, 10))
meq = 1

result = solve.QP(Dmat, dvec, Amat, bvec, meq)

w = result$solution 
w = (w / sum(w)) %>%
  round(., 4) %>%
  setNames(colnames(rets))

print(w)
