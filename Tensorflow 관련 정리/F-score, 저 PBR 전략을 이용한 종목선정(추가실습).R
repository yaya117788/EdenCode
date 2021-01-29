'''F-Score 지표는 조셉 피오트로스키 교수가 발표(Piotroski and others 2000)한 지표입니다. 
그는 논문에서, 저PBR을 이용한 밸류 전략은 높은 성과를 기록하지만 재무 상태가 불량한 기업이 많으며, 
저PBR 종목 중 재무적으로 우량한 기업을 선정해 투자한다면 성과를 훨씬 개선할 수 있다고 보았습니다.

F-Score에서는 재무적 우량 정도를 수익성(Profitability), 재무 성과(Financial Performance), 
운영 효율성(Operating Efficiency)으로 구분해 총 9개의 지표를 선정합니다.'''

## 수익성 > ROA, CFO, ROA의 증가율, ACCRUAL(CFO-ROA차이 이용)
## 재무성과 > 레버리지 증가\감소율 , 유동성증가\감소율, 발행주식수의 증가\감소 유무
## 운영효율성 > 매출총이익률이 증가율, 회전율 증가율

library(stringr)
library(ggplot2)
library(dplyr)

KOR_fs = readRDS('data/KOR_fs.Rds')
KOR_ticker = read.csv('data/KOR_ticker.csv', row.names = 1,
                      stringsAsFactors = FALSE) 
KOR_ticker$'종목코드' =
  str_pad(KOR_ticker$'종목코드', 6, 'left', 0)

## 먼저 티커데이터와 재무제표 데이터를 끌어온다.

''' 각 항목에 맞는 재무제표 데이터를 끌어와 각 지표를 구한다'''
# 수익성
ROA = KOR_fs$'지배주주순이익' / KOR_fs$'자산'
CFO = KOR_fs$'영업활동으로인한현금흐름' / KOR_fs$'자산'
ACCURUAL = CFO - ROA

# 재무성과
LEV = KOR_fs$'장기차입금' / KOR_fs$'자산'
LIQ = KOR_fs$'유동자산' / KOR_fs$'유동부채'
OFFER = KOR_fs$'유상증자'

# 운영 효율성
MARGIN = KOR_fs$'매출총이익' / KOR_fs$'매출액'
TURN = KOR_fs$'매출액' / KOR_fs$'자산'


# 위에서 구한 지표들을 이용해 F-score 점수부여 
if ( lubridate::month(Sys.Date()) %in% c(1,2,3,4) ) {
  num_col = str_which(colnames(KOR_fs[[1]]), as.character(lubridate::year(Sys.Date()) - 2))
} else {
  num_col = str_which(colnames(KOR_fs[[1]]), as.character(lubridate::year(Sys.Date()) - 1))
}
''' 1~4월에 데이터를 받으면 전년도 재무제표가 일부만 들어오는 경향이 있으므로, 전전년도 데이터를 사용해야 합니다.
그래서 1,2,3,4 월일시 전전년도 데이터를 사용하는 ifelse문'''

F_1 = as.integer(ROA[, num_col] > 0)
# ROA가 양수면 1점 
F_2 = as.integer(CFO[, num_col] > 0)
# CFO가 양수면 1점 
F_3 = as.integer(ROA[, num_col] - ROA[, (num_col-1)] > 0)
# ROA(총자산순이익률)이 증가했으면 1점
F_4 = as.integer(ACCURUAL[, num_col] > 0) 
# CFO- ROA를 해서 양수면 1점 
F_5 = as.integer(LEV[, num_col] - LEV[, (num_col-1)] <= 0) 
# 전년도에 비해 레버리지가 감소했으면 1점 (0 또는 음수가 나와야함)
F_6 = as.integer(LIQ[, num_col] - LIQ[, (num_col-1)] > 0)
# 전년도에 비해 유동성이 증가했으면 1점 
F_7 = as.integer(is.na(OFFER[,num_col]) |
                   OFFER[,num_col] <= 0)
# 유상증자 여부를 통해 없을시 1점부여 
F_8 = as.integer(MARGIN[, num_col] -
                   MARGIN[, (num_col-1)] > 0)
# 매출총이익을 매출액으로 나눈 매출총이익률이 증가했으면 1점

F_9 = as.integer(TURN[,num_col] - TURN[,(num_col-1)] > 0)
# 회전율이 증가했으면 1점 

F_Table = cbind(F_1, F_2, F_3, F_4, F_5, F_6, F_7, F_8, F_9) 
head(F_Table)

F_Score = F_Table %>%
  apply(., 1, sum, na.rm = TRUE) %>%
  setNames(KOR_ticker$`종목명`)
## 각각의 F-score들을 더해서 종목명을 이름으로 설정해 구분할 수 있게 한다.
(F_dist = prop.table(table(F_Score)) %>% round(3))
## 각 F-score 별로 분포가 어떻게 되는지 확인 > 주로 3~5에 분포되어있다.

F_dist %>%
  data.frame() %>%
  ggplot(aes(x = F_Score, y = Freq,
             label = paste0(Freq * 100, '%'))) +
  geom_bar(stat = 'identity') +
  geom_text(color = 'black', size = 3, vjust = -0.4) +
  scale_y_continuous(expand = c(0, 0, 0, 0.05),
                     labels = scales::percent) +
  ylab(NULL) +
  theme_classic() 
## 그래프로 확인해본다.


# 종목선정 
F_Score
invest_F_Score = F_Score %in% c(9)
KOR_ticker[invest_F_Score, ] %>% 
  select(`종목코드`, `종목명`) %>%
  mutate(`F-Score` = F_Score[invest_F_Score])



# 저 PBR 전략과 결합을 위한 저 PBR 종목선정
'''각 산업별로 PBR로 가치를 평가하는 기준이 다를수 있지만 대체적으로 1이하면 저평가라 
생각할 수 있어 PBR이 낮은 30개의 랭킹을 우선 구해본다.'''

KOR_value = read.csv('data/KOR_value.csv', row.names = 1,
                     stringsAsFactors = FALSE)

invest_pbr = rank(KOR_value$PBR) <= 30
KOR_ticker[invest_pbr, ] %>%
  select(`종목코드`, `종목명`) %>%
  mutate(`PBR` = round(KOR_value[invest_pbr, 'PBR'], 4))

intersect(KOR_ticker[invest_pbr, '종목명'],KOR_ticker[invest_F_Score,'종목명' ])

## F-score의 범위를 8까지 확대시켜본다.
invest_F_Score2 = F_Score %in% c(8,9)
intersect(KOR_ticker[invest_pbr, '종목명'],KOR_ticker[invest_F_Score2,'종목명' ])

## F-SCORE의 범위를 7까지 확대시켜본다.
invest_F_Score3 = F_Score %in% c(7,8,9)
intersect(KOR_ticker[invest_pbr, '종목명'],KOR_ticker[invest_F_Score3,'종목명' ])
