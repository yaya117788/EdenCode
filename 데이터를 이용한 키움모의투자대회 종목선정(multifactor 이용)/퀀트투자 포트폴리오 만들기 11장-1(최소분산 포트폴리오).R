library(quantmod)
library(PerformanceAnalytics)
library(magrittr)
library(tidyr)
library(dplyr)
library(corrplot)
library(nloptr)
# 글로벌 자산을 대표하는 ETF를 다운한다. (주식, 채권, 대체자산)
symbols = c('SPY', # 미국 주식
            'IEV', # 유럽 주식 
            'EWJ', # 일본 주식
            'EEM', # 이머징 주식
            'TLT', # 미국 장기채
            'IEF', # 미국 중기채
            'IYR', # 미국 리츠
            'RWX', # 글로벌 리츠
            'GLD', # 금
            'DBC'  # 상품
)
getSymbols(symbols, src = 'yahoo')

prices = do.call(cbind,
                 lapply(symbols, function(x) Ad(get(x)))) %>%
  setNames(symbols)
# Lapply를 이용해 Ad 즉 수정주가만을 선택하여 리스트로 뽑아와 do.call을 이용해 열로 묶는다.
rets = Return.calculate(prices) %>% na.omit()
rets
# Return method를 이용해 수익률을 계산하고 na값을 제외한다.

cor(rets) %>%
  corrplot(method = 'color', type = 'upper',
           addCoef.col = 'black', number.cex = 0.7,
           tl.cex = 0.6, tl.srt=45, tl.col = 'black',
           col =
             colorRampPalette(c('blue', 'white', 'red'))(200),
           mar = c(0,0,0.5,0))
''' 같은 자산군 내에서는 강한 상관관계를 보인다. 주식과 채권은 낮은 상관관계를 보이며, 주식과
리츠는 어느정도 높은 상관관계를 보인다. 금과 같은 대체자산들은 어떤자산과도 상관관계가 강하지 않다.'''



# 포트폴리오 최적화를 위해선 분산-공분산행렬이 가장 많이 사용되는데 cov method를 이용해 쉽게 구할 수 있다.
''' 이 행렬의 대각선 행렬 값은 각 변수의 분산이고, 대각선 이외의 행렬 값들은 변수 사이의 공분산이다. 
분산-공분산 행렬은 각 변수의 분산이 들어간 대각선을 중심으로 대칭행렬(symmetric matrix)을 이룬다.'''
covmat = cov(rets)


''' 최소분산 포트폴리오 구성방법
최소분산 포트폴리오 (MVP)는 최적화 작업을 통해 구할 수 있다.
일적으로 최적화는 크게
목적함수 (Objective function) 와 제약조건 (Constraint) 이 필요하다.
'''

### 먼저,  MVP의 목적함수에 대해 알아보면

#포트폴리오의 분산은 (1) 번과 같이 나타나며 (w'Ωw) MVP는 (2) 번과 같이 
# 이 (1)번의 값이 최소화되는 w를 찾는 최적화 작업을 통해 구할 수 있다.
# 그리고 투자비중이 음수가 나오면 공매도를 의미하는데 일반적으로 공매도는 불가하고 Long Only Potforlio를 위해 제약조건을 추가한다.
''' (3) 투자비중의 합이 1일 것 (100% 투자)
(4) 공매도 투자 (Short Position) 이 없을 것 이다.'''



# slsqp() 함수를 이용한 최적화
''' R에서 가장 손쉽게 최적화 작업을 수행하는 방법은 nloptr 패키지의 
slsqp() 함수를 이용하는 것입니다. slsqp() 함수는 순차적 이차 계획
(Sequential Quadratic Programming)을 이용해 해를 찾습니다.
목적함수는 min f(x) 최소화하고자 하는값 즉, 포트폴리오의 변동성이고 제약조건은
크게 개별자산의 투자비율이 0이상(b(x) >= 0)이고 합이 1인것입니다. (c(x)=0)인데 c(x)를 투자비율의 합 -1로 해서 넘기면 비중의 합이 1이된다.
'''

'''slsqp(x0, fn, gr = NULL, lower = NULL, upper = NULL,
      hin = NULL, hinjac = NULL, heq = NULL, heqjac = NULL,
      nl.info = FALSE, control = list(), ...)
      slsqp의 함수의 구성으로 우리가 구체적으로 입력해야할 값은 x0, fn, hin, heq 항목입니다.
      x0는 초기값으로 일반적으로 모든 x에 대해 동일한값을 입력,
      fn은 최소화하고자 하는 목적함수
      hin은 부등위 제약조건을 의미하며, 프로그래밍 내에서는 hin >=0으로 인식하여 각 자산의 비중이 0보다 크다는 제약조건과 연결
      heq는 등위 제약조건을 의미하며 heq ==0을 의마한다. 투자비중의합 -1의 형태를 입력하면 된다.
먼저 fn, hin, heq에 해당하는 함수를 각각 만든후 slsqp와 결합해 결과값을 얻는다.
'''

### 1. 목적함수 부분 (fn)
objective = function(w) {
  obj = t(w) %*% covmat %*% w
  return(obj)
}
''' covmat은 사전에 계산된 분산-공분산행렬이고 w는 각 자산의 투자비중입니다. obj는 즉, w′Ωw를 계산한 것입니다.'''

### 2. 부등위 제약조건 부분 (hin)
hin.objective = function(w) {
  return(w)
}
''' 패키지 내에서 알아서 >=0 형태로 입력하므로 w만 입력해주면 된다.'''

### 3. 등위 제약조건 부분 (heq)
heq.objective = function(w) {
  sum_w = sum(w)
  return( sum_w - 1 )
}
''' 먼저 계산된 비중인 w들의 합계를 구한 후 해당값에서 1을 빼주는 값을 반환하면 w의합 -1 = 0 즉 w의 합 = 1로 만들수 있다.'''

library(nloptr)

result = slsqp( x0 = rep(0.1, 10),
                fn = objective,
                hin = hin.objective,
                heq = heq.objective)
''' 예제로 종목이 10개가 있는데 우선 초기값으로 x0에는 동일한 비중들을 입력한다.(각 0.1) 이외에
fn, hin, heq parameter를 각각 만든 함수를 입력합니다.'''
## 초기값을 시작점으로 해 w값들을 조정하여 목적함수가 최소가 되는 지점의 w를 반환한다.
print(result$par)
# 최적화된 지점의 해를 표현한다. 각 자산들의 투자비중을 의미 
print(result$value)
# 결과값으로 나온 목적함수값이며 포트폴리오의 분산을 의미한다.

w_1 = result$par %>% round(., 4) %>%
  setNames(colnames(rets))

print(w_1)

# 반올림을 하고 각각의 값들에 맞는 투자자산의 이름을 입력한다. 계산된 비중으로 포트폴리오를 구성하면 포트폴리오의 비중이 최소가 됩니다.



# slove.QP() 함수를 이용한 최적화
''' 다음으로는 quadprog 패키지 내의 solve.QP() 함수를 이용해 포트폴리오 최적화를 하는 방법이 있습니다. 
해당 함수는 쌍대기법(Dual Method)을 이용해 제약조건 내에서 목적함수가 최소화되는 해를 구합니다.
min(−(d^T)b+1/2(b^T)Db)이 목적함수이고 제약조건은 (A^T)b >= b0로 나타내어 진다.
목적함수부분은 매우 이해하기 쉽게 되어있는데, b를 각 투자비중인 w, D를 분산-공분산 행렬인 Ω, d= 0 으로 하면 최소분산 pot의 목적함수와 동일해진다.
제약조건부분은 A^T부분을 잘 수정하면 두개의 필요제약조건을 만들 수 있다.
'''

''' # solve.QP(Dmat, dvec, Amat, bvec, meq = 0, factorized = FALSE)
이런 형태로 구성되어있고 각 
Dmat은 목적함수중 D에 해당하는 행렬부분으로서 분산-공분산 행렬과 일치한다.
dvec은 목적함수 중 d에 해당하는 벡터 부분이며, 포트폴리오 최적화에서는 역할이 없다.
Amat은 제약조건중 A^T에 해당하는 부분으로써, 제약조건중 좌변에 위치하는 항목 제약조건 행렬을 구한 후 이것의 전치행렬을 입력해야하는데 주의하기
bvec은 제약조건 중 b0에 해당하는 부분으로써, 제약조건 중 우변에 위치하는 항목
meq은 bvec의 몇번째까지를 등위 제약조건으로 설정할지에 대한 부분이다.
solve.Qp()함수를 이용할땐 A^T 항목을 제대로 입력하는 것이 가장 중요하며 나머지항목은 손쉽게 입력가능.
'''
Dmat = covmat # 분산 공분산 행렬을 입력한다. 
dvec = rep(0, 10) # 필요한값이 아니니 0벡터를 입력
Amat = t(rbind(rep(1, 10), diag(10), -diag(10)))
bvec = c(1, rep(0, 10), -rep(1, 10))
meq = 1
'''제약조건은 크게 투자비중의 합이 1인 제약조건, 최소 투자비중이 0 이상인 제약조건, 
최대 투자비중이 1 이하인 제약조건, 총 세 개 부분으로 나눌 수 있습니다. solve.Qp() 함수의 제약조건은 
항상 좌변이 큰 형태이므로, 최대투자비중에 대한 제약조건은 양변에 (-)를 곱해 부등호를 맞춰준다.
meq =1로 설정해 첫번째 제약조건은 등식제약조건임을 선언한다.
Amat 과정을 만드는 과정은 먼저 rep(1,10)을 통해 최상단에 위치한 1로 이루어진 행렬을 만들어준다.
이후 하단의 1과 -1로 이루어진 대각행렬은 diag()함수를 통해 쉽게 만들 수 있다.
ex> diag(3)으로 할경우 3x3대각행렬을 -diag(3)을 할경우 대각이 -1인 3x3 대각행렬이 만들어진다.
이후 rbind를 통해 세개의 행렬을 행으로 묶어주고 Transpose 하면 Amat에 입력하면 된다.
'''

library(quadprog)
result = solve.QP(Dmat, dvec, Amat, bvec, meq)

print(result$solution)
# solution은 위의 par와 같이 각 자산의 투자비중을 의미 
print(result$value)
# value는 포트폴리오 분산을 의미 = 목적함수의 결과값
w_2 = result$solution %>% round(., 4) %>%
  setNames(colnames(rets))

print(w_2)


# optimalPortfolio() 함수를 이용한 최적화
'''RiskPortfolios 패키지의 optimalPortfolio() 함수를 이용해 매우 간단하게 최적화 포트폴리오를 구현할 수도 있습니다.
#optimalPortfolio(Sigma, mu = NULL, semiDev = NULL,
                 control = list())
의 형식으로 되어있으며 sigma는 분산-공분산행렬,
mu와 semiDev는 각각 기대수익률과 세미편차(Semi deviation)로서 입력하지 않아도 된다.
control은 포트폴리오의 종류 및 제약조건에 해당하는 부분으로 여러 인자가 있다.
control 항목에서 원하는 포트폴리오 타입(type)과 제약조건(constraint)를 입력해주면 매우 손쉽게 구성가능
1. type : minvol(최소분산) invvol(역변동성), erc(위험균형), maxdiv(최대분산효과), riskeff(위험-효율적)
2. constraint : lo(최소투자비중이 0보다 클것), user(최소 및 최대투자비중 설정)
'''
library(RiskPortfolios)

w_3 = optimalPortfolio(covmat,
                       control = list(type = 'minvol',
                                      constraint = 'lo')) %>%
  round(., 4) %>%
  setNames(colnames(rets))
###  비중의 합이 1인 제약조건은 자동 적용된다.
print(w_3)  




# 위 세 함수 slsqp(), solve.QP(), optimalPortfolio()를 이용하여 구한 값들의 비교
library(ggplot2)

data.frame(w_1) %>%
  ggplot(aes(x = factor(rownames(.), levels = rownames(.)),
             y = w_1)) +
  geom_col() +
  xlab(NULL) + ylab(NULL)

''' 세가지 방법 모두 결과가 동일하다. 하지만 여기서 나온 결과를 보면 대부분 투자비중이 0%이고 특정자산 예로 IEF에 79.27%를 투자하는
편중된 결과가 나온다. 이를 "구석해문제(Corner Solution)"이라 하고 이 문제를 해결하기 위해
각 자산의 최소 및 최대 투자비중 제약조건을 추가해주면 된다.'''



# 최소 및 최대 투자비중 제약조건
'''모든 자산에 골고루 투자하기 위해 개별 투자비중을 최소 5% 최대 20%로 하는 제약조건을 추가하겠습니다.'''
## slsqp() 함수에 제약조건을 추가
result = slsqp( x0 = rep(0.1, 10),
                fn = objective,
                hin = hin.objective,
                heq = heq.objective,
                lower = rep(0.05, 10),
                upper = rep(0.20, 10))
### 기존 사용했던 제약조건 외에 lower, upper 제약조건을 추가하여 그 사이에 해를 갖게 한다.
w_4 = result$par %>% round(., 4) %>%
  setNames(colnames(rets))

print(w_4)

## solve.Qp() 함수내에서 제약조건을 추가하는 법 
''' 해당 함수역시 다른입력값은 동일하며 제약조건의 우변에 해당하는 bvec항목만 수정하면 된다. 기존
[0,1]에서 [0.05,0.20]으로 변경하면 bvec행렬은 [1,0.05,0.05 ```` (-)0.20,-0.20 ]이런식으로 된다.'''
Dmat = covmat
dvec = rep(0, 10)
Amat = t(rbind(rep(1, 10), diag(10), -diag(10)))
bvec = c(1, rep(0.05, 10), -rep(0.20, 10))
meq = 1

result = solve.QP(Dmat, dvec, Amat, bvec, meq)

w_5 = result$solution %>% round(., 4) %>%
  setNames(colnames(rets))

print(w_5)

## optimalPortfolio() 함수에서의 조건 추가
''' control 항목중 constraint 부분을 간단하게 수정해 원하는 조건을 입력할 수 있습니다.'''
w_6 = optimalPortfolio(covmat,
                       control = list(type = 'minvol',
                                      constraint = 'user',
                                      LB = rep(0.05, 10),
                                      UB = rep(0.20, 10))) %>%
  round(., 4) %>%
  setNames(colnames(rets))
### LB, UB(각각 lower bound, upper bound) Control 항목이 추가되었다. LB에 최소투자비중 벡터를, UB에는 최대투자비중 벡터를 입력합니다.
print(w_6)

data.frame(w_4) %>%
  ggplot(aes(x = factor(rownames(.), levels = rownames(.)),
             y = w_4)) +
  geom_col() +
  geom_hline(aes(yintercept = 0.05), color = 'red') +
  geom_hline(aes(yintercept = 0.20), color = 'red') +
  xlab(NULL) + ylab(NULL)
## 어느정도 해결된 것을 그래프를 통해 확인할 수 있다.



# 각 자산별 제약조건의 추가
''' 투자 규모가 크지 않다면 위에서 추가한 제약조건만으로도 충분히 훌륭한 포트폴리오가 구성됩니다.
그러나 투자 규모가 커지면 추가적인 제약조건들을 고려해야 할 경우가 생깁니다. 
예를 들어 벤치마크 비중과의 괴리로 인한 추적오차(Tracking Error)를 고려해야 할 수도 있고, 투자대상별 거래량을 고려한 제약조건을 추가해야 할 때가 있다.
자산별로 상이한 제약조건이 필요하다.
slsqp()와 optimal~()함수는 복잡한 제약조건을 다루기 힘들지만 solve.Qp()함수는 bvec부분을 간단하게 수정해
어렵지 않게 구현이 가능하다.'''


## 각 최소, 최대투자비중을 다르게 조정해본다. 
'''각 최소비중이 0.1,0.1,0.05,0.05,0.1,0.1,0.05,0.05,0.03,0.03으로하고
   각 최대비중을 0.25,0.25,0.2,0.2,0.2,0.2,0.1,0.1,0.08,0.08 로 설정한다.
그럼 행렬이 [1,0.1,0.1,0.05``` -0.25,-0.25``] 이런식으로 된다.'''

Dmat = covmat
dvec = rep(0, 10)
Amat = t(rbind(rep(1, 10), diag(10), -diag(10))) 
bvec = c(1, c(0.10, 0.10, 0.05, 0.05, 0.10,
              0.10, 0.05, 0.05, 0.03, 0.03),
         -c(0.25, 0.25, 0.20, 0.20, 0.20,
            0.20, 0.10, 0.10, 0.08, 0.08))
meq = 1
## 각각의 최소,최대 자산비중을 입력하여 벡터를 구성한다.
result = solve.QP(Dmat, dvec, Amat, bvec, meq)

result$solution %>%
  round(., 4) %>%
  setNames(colnames(rets))


